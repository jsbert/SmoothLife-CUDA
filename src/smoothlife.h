#ifndef SMOOTHLIFE_SMOOTHLIFE_H_
#define SMOOTHLIFE_SMOOTHLIFE_H_
#include <cublas_v2.h>
#include <cuda.h>
#include <cufft.h>

class SmoothLife {
 public:
  SmoothLife(int len_x, int len_y, double dt);
  SmoothLife(SmoothLife&) = delete;
  SmoothLife& operator=(const SmoothLife&) = delete;
  SmoothLife(SmoothLife&&) = delete;
  SmoothLife& operator=(SmoothLife&&) = delete;
  ~SmoothLife();

  CUdeviceptr frame() const { return frame_; }
  void Step();

 private:
  void InitMN();
  void InitWorld();
  void RunCreateAntialiasedCircle(CUdeviceptr buf, double radius);
  void RunSubtractD(CUdeviceptr buf1, CUdeviceptr buf2);
  void RunDivideScalarD(CUdeviceptr buf, double val);
  void RunMultiplyZ(CUdeviceptr buf1, CUdeviceptr buf2);
  void RunS(double dt, CUdeviceptr world, CUdeviceptr _M, CUdeviceptr _N);
  void RunPaint(CUdeviceptr world, CUdeviceptr frame);

  int len_x_;
  int len_y_;
  double dt_;
  CUmodule mod_;
  cublasHandle_t cublas_;
  cufftHandle cufft_D2Z_;
  cufftHandle cufft_Z2D_;

  CUfunction k_create_antialiased_circle_;
  CUfunction k_subtract_D_;
  CUfunction k_divide_scalar_D_;
  CUfunction k_multiply_Z_;
  CUfunction k_s_;
  CUfunction k_paint_;

  CUdeviceptr frame_;
  CUdeviceptr world_;
  CUdeviceptr M_;
  CUdeviceptr N_;
  CUdeviceptr M_tmp_;
  CUdeviceptr N_tmp_;
  CUdeviceptr _M_;
  CUdeviceptr _N_;
};

#endif  // SMOOTHLIFE_SMOOTHLIFE_H_
