#include "smoothlife.h"

#include <cublas_v2.h>
#include <cuda.h>
#include <cufft.h>

#include <cstdint>
#include <cstring>
#include <random>

#include "util.h"

namespace {
uint8_t fatbin[] = {
#embed FATBIN_FILE
};
}  // namespace

SmoothLife::SmoothLife(int len_x, int len_y, double dt)
    : len_x_(len_x), len_y_(len_y), dt_(dt) {
  CU_SUCCEED(cuModuleLoadData(&mod_, fatbin));

  CUBLAS_SUCCEED(cublasCreate(&cublas_));

  CUFFT_SUCCEED(cufftPlan2d(&cufft_D2Z_, len_y_, len_x_, CUFFT_D2Z));
  CUFFT_SUCCEED(cufftPlan2d(&cufft_Z2D_, len_y_, len_x_, CUFFT_Z2D));

  CU_SUCCEED(cuModuleGetFunction(&k_create_antialiased_circle_, mod_,
                                 "CreateAntialiasedCircle"));
  CU_SUCCEED(cuModuleGetFunction(&k_subtract_D_, mod_, "SubtractD"));
  CU_SUCCEED(cuModuleGetFunction(&k_divide_scalar_D_, mod_, "DivideScalarD"));
  CU_SUCCEED(cuModuleGetFunction(&k_multiply_Z_, mod_, "MultiplyZ"));
  CU_SUCCEED(cuModuleGetFunction(&k_s_, mod_, "S"));
  CU_SUCCEED(cuModuleGetFunction(&k_paint_, mod_, "Paint"));

  CU_SUCCEED(cuMemAlloc(&frame_, len_x_ * len_y_ * sizeof(uint32_t)));
  CU_SUCCEED(cuMemAlloc(&world_, len_x_ * len_y_ * sizeof(double)));
  CU_SUCCEED(cuMemAlloc(&M_, len_x_ * len_y_ * sizeof(cufftDoubleComplex)));
  CU_SUCCEED(cuMemAlloc(&N_, len_x_ * len_y_ * sizeof(cufftDoubleComplex)));
  CU_SUCCEED(cuMemAlloc(&M_tmp_, len_x_ * len_y_ * sizeof(cufftDoubleComplex)));
  CU_SUCCEED(cuMemAlloc(&N_tmp_, len_x_ * len_y_ * sizeof(cufftDoubleComplex)));
  CU_SUCCEED(cuMemAlloc(&_M_, len_x_ * len_y_ * sizeof(double)));
  CU_SUCCEED(cuMemAlloc(&_N_, len_x_ * len_y_ * sizeof(double)));

  InitMN();
  InitWorld();
  RunPaint(world_, frame_);
}

SmoothLife::~SmoothLife() {
  CU_SUCCEED(cuMemFree(M_));
  CU_SUCCEED(cuMemFree(N_));
  CU_SUCCEED(cuMemFree(M_tmp_));
  CU_SUCCEED(cuMemFree(N_tmp_));
  CU_SUCCEED(cuMemFree(_M_));
  CU_SUCCEED(cuMemFree(_N_));
  CU_SUCCEED(cuMemFree(world_));
  CU_SUCCEED(cuMemFree(frame_));

  CUFFT_SUCCEED(cufftDestroy(cufft_Z2D_));
  CUFFT_SUCCEED(cufftDestroy(cufft_D2Z_));

  CUBLAS_SUCCEED(cublasDestroy(cublas_));

  CU_SUCCEED(cuModuleUnload(mod_));
}

void SmoothLife::InitMN() {
  CUdeviceptr inner_circle, outer_circle;
  double inner_radius = 7.0;
  double outer_radius = inner_radius * 3.0;

  CU_SUCCEED(cuMemAlloc(&inner_circle, len_x_ * len_y_ * sizeof(double)));
  CU_SUCCEED(cuMemAlloc(&outer_circle, len_x_ * len_y_ * sizeof(double)));

  RunCreateAntialiasedCircle(inner_circle, inner_radius);
  RunCreateAntialiasedCircle(outer_circle, outer_radius);

  // Subtract inner_circle from outer_circle
  RunSubtractD(outer_circle, inner_circle);

  // Scale buffers so the sum is 1. Using asum here is ok, there are no
  // negative numbers.
  double sum;
  CUBLAS_SUCCEED(cublasDasum(cublas_, len_x_ * len_y_,
                             reinterpret_cast<double*>(inner_circle), 1, &sum));
  RunDivideScalarD(inner_circle, sum);

  CUBLAS_SUCCEED(cublasDasum(cublas_, len_x_ * len_y_,
                             reinterpret_cast<double*>(outer_circle), 1, &sum));
  RunDivideScalarD(outer_circle, sum);

  CUFFT_SUCCEED(cufftExecD2Z(cufft_D2Z_,
                             reinterpret_cast<cufftDoubleReal*>(inner_circle),
                             reinterpret_cast<cufftDoubleComplex*>(M_)));
  CUFFT_SUCCEED(cufftExecD2Z(cufft_D2Z_,
                             reinterpret_cast<cufftDoubleReal*>(outer_circle),
                             reinterpret_cast<cufftDoubleComplex*>(N_)));

  CU_SUCCEED(cuMemFree(inner_circle));
  CU_SUCCEED(cuMemFree(outer_circle));
}

void SmoothLife::InitWorld() {
  double* new_world = new double[len_x_ * len_y_];
  memset(new_world, 0, len_x_ * len_y_ * sizeof(double));

  int num_squares = (len_x_ * len_y_) / 3000;
  int square_dim = 20;
  std::random_device r;
  std::default_random_engine e(r());
  std::uniform_int_distribution<int> uniform_dist_x(0, len_x_ - 1);
  std::uniform_int_distribution<int> uniform_dist_y(0, len_y_ - 1);

  for (int i = 0; i < num_squares; ++i) {
    int start_x = uniform_dist_x(e);
    int end_x = start_x + square_dim;
    if (end_x > len_x_) {
      end_x = len_x_;
    }

    int start_y = uniform_dist_y(e);
    int end_y = start_y + square_dim;
    if (end_y > len_y_) {
      end_y = len_y_;
    }

    for (int y = start_y; y < end_y; ++y) {
      for (int x = start_x; x < end_x; ++x) {
        new_world[y * len_x_ + x] = 1.0;
      }
    }
  }

  CU_SUCCEED(cuMemcpyHtoD(world_, new_world, len_x_ * len_y_ * sizeof(double)));
  delete[] new_world;
}

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

void SmoothLife::RunCreateAntialiasedCircle(CUdeviceptr buf, double radius) {
  void* args[] = {&len_x_, &len_y_, &buf, &radius};
  unsigned int num_blocks_x = CEIL_DIV(len_x_, BLOCK_SIZE_X);
  unsigned int num_blocks_y = CEIL_DIV(len_y_, BLOCK_SIZE_Y);
  CU_SUCCEED(cuLaunchKernel(k_create_antialiased_circle_, num_blocks_x,
                            num_blocks_y, 1, BLOCK_SIZE_X, BLOCK_SIZE_Y, 1, 0,
                            nullptr, args, nullptr));
}

void SmoothLife::RunSubtractD(CUdeviceptr buf1, CUdeviceptr buf2) {
  void* args[] = {&len_x_, &len_y_, &buf1, &buf2};
  unsigned int num_blocks_x = CEIL_DIV(len_x_, BLOCK_SIZE_X);
  unsigned int num_blocks_y = CEIL_DIV(len_y_, BLOCK_SIZE_Y);
  CU_SUCCEED(cuLaunchKernel(k_subtract_D_, num_blocks_x, num_blocks_y, 1,
                            BLOCK_SIZE_X, BLOCK_SIZE_Y, 1, 0, nullptr, args,
                            nullptr));
}

void SmoothLife::RunDivideScalarD(CUdeviceptr buf, double val) {
  void* args[] = {&len_x_, &len_y_, &buf, &val};
  unsigned int num_blocks_x = CEIL_DIV(len_x_, BLOCK_SIZE_X);
  unsigned int num_blocks_y = CEIL_DIV(len_y_, BLOCK_SIZE_Y);
  CU_SUCCEED(cuLaunchKernel(k_divide_scalar_D_, num_blocks_x, num_blocks_y, 1,
                            BLOCK_SIZE_X, BLOCK_SIZE_Y, 1, 0, nullptr, args,
                            nullptr));
}

void SmoothLife::RunMultiplyZ(CUdeviceptr buf1, CUdeviceptr buf2) {
  void* args[] = {&len_x_, &len_y_, &buf1, &buf2};
  unsigned int num_blocks_x = CEIL_DIV(len_x_, BLOCK_SIZE_X);
  unsigned int num_blocks_y = CEIL_DIV(len_y_, BLOCK_SIZE_Y);
  CU_SUCCEED(cuLaunchKernel(k_multiply_Z_, num_blocks_x, num_blocks_y, 1,
                            BLOCK_SIZE_X, BLOCK_SIZE_Y, 1, 0, nullptr, args,
                            nullptr));
}

void SmoothLife::RunS(double dt, CUdeviceptr world, CUdeviceptr _M,
                      CUdeviceptr _N) {
  void* args[] = {&len_x_, &len_y_, &dt, &world, &_M, &_N};
  unsigned int num_blocks_x = CEIL_DIV(len_x_, BLOCK_SIZE_X);
  unsigned int num_blocks_y = CEIL_DIV(len_y_, BLOCK_SIZE_Y);
  CU_SUCCEED(cuLaunchKernel(k_s_, num_blocks_x, num_blocks_y, 1, BLOCK_SIZE_X,
                            BLOCK_SIZE_Y, 1, 0, nullptr, args, nullptr));
}

void SmoothLife::RunPaint(CUdeviceptr world, CUdeviceptr frame) {
  void* args[] = {&len_x_, &len_y_, &world, &frame};
  unsigned int num_blocks_x = CEIL_DIV(len_x_, BLOCK_SIZE_X);
  unsigned int num_blocks_y = CEIL_DIV(len_y_, BLOCK_SIZE_Y);
  CU_SUCCEED(cuLaunchKernel(k_paint_, num_blocks_x, num_blocks_y, 1,
                            BLOCK_SIZE_X, BLOCK_SIZE_Y, 1, 0, nullptr, args,
                            nullptr));
}

void SmoothLife::Step() {
  // M_tmp_ = fft2(world_)
  // N_tmp_ = M_tmp_
  CUFFT_SUCCEED(cufftExecD2Z(cufft_D2Z_,
                             reinterpret_cast<cufftDoubleReal*>(world_),
                             reinterpret_cast<cufftDoubleComplex*>(M_tmp_)));
  CU_SUCCEED(cuMemcpyDtoD(N_tmp_, M_tmp_,
                          len_x_ * len_y_ * sizeof(cufftDoubleComplex)));

  // M_tmp_ *= M_
  // N_tmp_ *= N_
  RunMultiplyZ(M_tmp_, M_);
  RunMultiplyZ(N_tmp_, N_);

  // _M_ = real(ifft2(M_tmp_))
  // _N_ = real(ifft2(N_tmp_))
  CUFFT_SUCCEED(cufftExecZ2D(cufft_Z2D_,
                             reinterpret_cast<cufftDoubleComplex*>(M_tmp_),
                             reinterpret_cast<cufftDoubleReal*>(_M_)));
  CUFFT_SUCCEED(cufftExecZ2D(cufft_Z2D_,
                             reinterpret_cast<cufftDoubleComplex*>(N_tmp_),
                             reinterpret_cast<cufftDoubleReal*>(_N_)));
  double num_elems = len_x_ * len_y_;
  RunDivideScalarD(_M_, num_elems);
  RunDivideScalarD(_N_, num_elems);

  // s(world_, _M_, _N_) (updates world)
  RunS(dt_, world_, _M_, _N_);

  RunPaint(world_, frame_);
}
