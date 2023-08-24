﻿#include <stdio.h>	// printf
#include <stdint.h>	// uint32_t
#include <vector>	// std::vector

#include <cuda_runtime.h>

// reminder that GpuMemoryManager is in bytes

void FailIf(bool condition, const char* message)
{
	if (condition)
	{
		fprintf(stderr, "%s", message);
		exit(0);
	}
}

struct GpuMemoryManager
{
	struct MemoryData
	{
		float* address;
		size_t size;
		size_t dynamicSize;
		float ratio;
	};

	struct TensorData
	{
		float** address;
		size_t size;
		float ratio;
		MemoryData* memoryPtr;
	};

	std::vector<MemoryData*> availableMemory;
	std::vector<TensorData*> dynamicTensors;
	std::vector<TensorData*> staticTensors;

	GpuMemoryManager()
	{
		printf("Initializing GPU memory manager...\n\n");
		size_t freeMem, totalMem;
		cudaMemGetInfo(&freeMem, &totalMem);

		MemoryData* memData;
		size_t low, high, guess;
		cudaError_t err;
		do
		{
			memData = new MemoryData;
			low = 1, high = freeMem;
			do
			{
				guess = (low + high) >> 1;
				err = cudaMalloc((void**)&memData->address, guess);
				err == cudaSuccess ? low = guess + 1 : high = guess - 1;
				cudaFree(memData->address);
			} while (low <= high);
			low--;

			if (low > 0)
			{
				memData->size = low;
				cudaMalloc((void**)&memData->address, low);
				memData->dynamicSize = 0;
				freeMem -= low;
				availableMemory.emplace_back(memData);
			}
		} while (low > 0);

		FailIf(availableMemory.size() <= 0, "No available memory\n");
	}

	~GpuMemoryManager()
	{
		for (auto& memoryPtr : availableMemory)
		{
			cudaFree(memoryPtr->address);
			delete memoryPtr;
		}
	}

	void ManageStatic(float** tensorPtr, size_t size)
	{
		TensorData* tensorData = new TensorData;
		tensorData->address = tensorPtr;
		tensorData->size = size;
		staticTensors.emplace_back(tensorData);
		FailIf(tensorData->size <= 0, "Static Tensor size is <= 0\n");
	}

	void ManageDynamic(float** tensorPtr, size_t size)
	{
		TensorData* tensorData = new TensorData;
		tensorData->address = tensorPtr;
		tensorData->size = size;
		dynamicTensors.emplace_back(tensorData);
		FailIf(tensorData->size <= 0, "Dynamic Tensor size is <= 0\n");
	}

	void allocateStatic(uint32_t tensorIdx, float& largestRatio, std::vector<MemoryData*>& bestCombination, size_t& largestN)
	{
		if (tensorIdx == staticTensors.size())
			allocateDynamic(0, largestRatio, bestCombination, largestN);
		else
			for (MemoryData* memoryPtr : availableMemory)
				if (memoryPtr->size >= staticTensors[tensorIdx]->size)
				{
					staticTensors[tensorIdx]->memoryPtr = memoryPtr;
					memoryPtr->ratio -= staticTensors[tensorIdx]->ratio;
					memoryPtr->size -= staticTensors[tensorIdx]->size;
					allocateStatic(tensorIdx + 1, largestRatio, bestCombination, largestN);
					memoryPtr->ratio += staticTensors[tensorIdx]->ratio;
					memoryPtr->size += staticTensors[tensorIdx]->size;
				}
	}

	void allocateDynamic(uint32_t tensorIdx, float& largestRatio, std::vector<MemoryData*>& bestCombination, size_t& largestN)
	{
		if (tensorIdx == dynamicTensors.size())
		{
			float smallestRatio = 1;
			size_t size = 0;
			size_t dynamicSize = 0;
			for (auto& memoryPtr : availableMemory)
				if (memoryPtr->dynamicSize > 0 && memoryPtr->ratio < smallestRatio)
				{
					smallestRatio = memoryPtr->ratio;
					size = memoryPtr->size;
					dynamicSize = memoryPtr->dynamicSize;
				}
			printf("Smallest ratio: %f\n", smallestRatio);
			printf("Largest ratio: %f\n", largestRatio);
			printf("size: %zu\n", size);
			printf("dynamicSize: %zu\n\n", dynamicSize);

			if (smallestRatio > largestRatio)
			{
				printf("New best ratio: %f\n\n", smallestRatio);
				if (dynamicSize > 0)
					largestN = size / dynamicSize;
				largestRatio = smallestRatio;

				printf("largestN: %zu\n\n", largestN);
				printf("left over: %zu\n\n", size - largestN * dynamicSize);

				for (int i = 0; i < staticTensors.size(); ++i)
					bestCombination[i] = staticTensors[i]->memoryPtr;
				for (int i = 0; i < dynamicTensors.size(); ++i)
					bestCombination[i + staticTensors.size()] = dynamicTensors[i]->memoryPtr;
			}
		}
		else
			for (MemoryData* memoryPtr : availableMemory)
			{
				dynamicTensors[tensorIdx]->memoryPtr = memoryPtr;
				memoryPtr->ratio -= dynamicTensors[tensorIdx]->ratio;
				memoryPtr->dynamicSize += dynamicTensors[tensorIdx]->size;
				allocateDynamic(tensorIdx + 1, largestRatio, bestCombination, largestN);
				memoryPtr->ratio += dynamicTensors[tensorIdx]->ratio;
				memoryPtr->dynamicSize -= dynamicTensors[tensorIdx]->size;
			}
	}

	void Allocate(size_t& largestN)
	{
		size_t fragSize = 0;
		size_t dynamicTensorSize = 0;

		for (auto& memoryPtr : availableMemory)
			fragSize += memoryPtr->size;
		for (auto& tensor : staticTensors)
		{
			FailIf(tensor->size > fragSize, "Static tensor size is larger than total memory size\n");
			fragSize -= tensor->size;
		}
		for (auto& tensor : dynamicTensors)
			dynamicTensorSize += tensor->size;

		for (auto& memoryPtr : availableMemory)
			memoryPtr->ratio = (float)memoryPtr->size / fragSize;
		for (auto& tensor : staticTensors)
			tensor->ratio = (float)tensor->size / fragSize;
		for (auto& tensor : dynamicTensors)
			tensor->ratio = (float)tensor->size / dynamicTensorSize;

		largestN = 0;
		float largestRatio = -1;
		std::vector<MemoryData*> bestCombination(staticTensors.size() + dynamicTensors.size());
		allocateStatic(0, largestRatio, bestCombination, largestN);

		FailIf(bestCombination[0] == nullptr, "No combination found\n");

		// allocate memory
		for (auto& memoryPtr : availableMemory)
			memoryPtr->dynamicSize = 0;

		for (int i = 0; i < staticTensors.size(); ++i)
		{
			MemoryData* memoryPtr = bestCombination[i];
			*staticTensors[i]->address = memoryPtr->address + memoryPtr->dynamicSize;
			memoryPtr->dynamicSize += staticTensors[i]->size;
		}

		for (int i = 0; i < dynamicTensors.size(); ++i)
		{
			MemoryData* memoryPtr = bestCombination[i + staticTensors.size()];
			*dynamicTensors[i]->address = memoryPtr->address + memoryPtr->dynamicSize;
			memoryPtr->dynamicSize += dynamicTensors[i]->size * largestN;
		}

		// clean up
		for (auto& tensor : dynamicTensors)
			delete tensor;
		for (auto& tensor : staticTensors)
			delete tensor;

		dynamicTensors.clear();
		staticTensors.clear();
	}

	void PrintGpuMem() const
	{
		for (MemoryData* frag : availableMemory)
			printf("Allocated %zu bytes at %p\n", frag->size, frag->address);
		printf("\n");
	}

	void PrintMemory() const
	{
		// code below doesn't work cuz in gpu
		/*for (auto& memoryPtr : availableMemory)
		{
			for (int i = 0; i < memoryPtr->size; ++i)
				printf("%1.0f ", memoryPtr->address[i]);
			printf("\n\n");
		}*/
	}
};

int main()
{
	GpuMemoryManager gpuMemoryManager;

	size_t batches;
	float* staticArr1, * staticArr2, * dynamicArr1, * dynamicArr2;
	size_t sSize1 = 10;
	size_t sSize2 = 120;
	size_t dCoef1 = 3;
	size_t dCoef2 = 5;

	gpuMemoryManager.ManageStatic(&staticArr1, sSize1 * sizeof(float));
	gpuMemoryManager.ManageStatic(&staticArr2, sSize2 * sizeof(float));
	gpuMemoryManager.ManageDynamic(&dynamicArr1, dCoef1 * sizeof(float));
	gpuMemoryManager.ManageDynamic(&dynamicArr2, dCoef2 * sizeof(float));

	gpuMemoryManager.Allocate(batches);
	printf("batches: %zu\n\n", batches);

	size_t dSize1 = dCoef1 * batches;
	size_t dSize2 = dCoef2 * batches;
	float* hstaticArr1, * hstaticArr2, * hdynamicArr1, * hdynamicArr2;

	hstaticArr1 = new float[sSize1];
	hstaticArr2 = new float[sSize2];
	hdynamicArr1 = new float[dSize1];
	hdynamicArr2 = new float[dSize2];

	for (size_t i = 0; i < sSize1; ++i)
		hstaticArr1[i] = i;
	for (size_t i = 0; i < sSize2; ++i)
		hstaticArr2[i] = i;
	for (size_t i = 0; i < dSize1; ++i)
		hdynamicArr1[i] = i;
	for (size_t i = 0; i < dSize2; ++i)
		hdynamicArr2[i] = i;

	cudaMemcpy(staticArr1, hstaticArr1, sSize1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(staticArr2, hstaticArr2, sSize2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dynamicArr1, hdynamicArr1, dSize1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dynamicArr2, hdynamicArr2, dSize2 * sizeof(float), cudaMemcpyHostToDevice);

	// error check

	memset(hstaticArr1, 0, sSize1 * sizeof(float));
	memset(hstaticArr2, 0, sSize2 * sizeof(float));
	memset(hdynamicArr1, 0, dSize1 * sizeof(float));
	memset(hdynamicArr2, 0, dSize2 * sizeof(float));

	cudaMemcpy(hstaticArr1, staticArr1, sSize1 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hstaticArr2, staticArr2, sSize2 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hdynamicArr1, dynamicArr1, dSize1 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hdynamicArr2, dynamicArr2, dSize2 * sizeof(float), cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < sSize1; ++i)
		FailIf(hstaticArr1[i] != i, "Error in staticArr1\n");
	for (size_t i = 0; i < sSize2; ++i)
		FailIf(hstaticArr2[i] != i, "Error in staticArr2\n");
	for (size_t i = 0; i < dSize1; ++i)
		FailIf(hdynamicArr1[i] != i, "Error in dynamicArr1\n");
	for (size_t i = 0; i < dSize2; ++i)
		FailIf(hdynamicArr2[i] != i, "Error in dynamicArr2\n");

	gpuMemoryManager.PrintGpuMem();

	return 0;
}