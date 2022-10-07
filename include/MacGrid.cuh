#ifndef MAC_GRID_CUH
#define MAC_GRID_CUH

#include <iostream>

#include "Utils/OpenGL.h"

class MacGrid {

public:
  MacGrid(int _gridSize_i, int _gridSize_j, int _gridSize_k)
      : gridSize_i(_gridSize_i), gridSize_j(_gridSize_j),
        gridSize_k(_gridSize_k), cellNum(gridSize_i * gridSize_j * gridSize_k) {
    std::cout << cellNum << std::endl;
    cudaMalloc(&d_pressureBuff, cellNum * sizeof(float));
  }

  void getPressure(float *h_pressureBuff) {
    int error = cudaMemcpy(h_pressureBuff, d_pressureBuff,
                           cellNum * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << error << std::endl;
  }

  // Grid size expressed in number of cells in each dimension

  int gridSize_i;
  int gridSize_j;
  int gridSize_k;

  // Total number of cells

  int cellNum;

  float *d_pressureBuff;

  // Cell center index
  int ccIdx(int i, int j, int k) const {
    return k * gridSize_j * gridSize_i + j * gridSize_i + i;
  }
};

__global__ void initPressure(MacGrid grid, int newPressure) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < grid.cellNum)
    grid.d_pressureBuff[idx] = newPressure;
}

void gridPressureTest() {

  MacGrid grid(10, 10, 10);

  initPressure<<<1, grid.cellNum>>>(grid, 5);
  // initPressureArray<<<0, grid.cellNum>>>(grid.d_pressureBuff, grid.cellNum,
  // 5);

  float *h_array = (float *)malloc(grid.cellNum * sizeof(float));

  grid.getPressure(h_array);

  for (int i = 0; i < grid.cellNum; i++) {
    printf("h_array[%d]: %f\n", i, h_array[i]);
  }
  return;
}

#endif /* MAC_GRID_CUH */