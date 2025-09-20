#include <cstdint>

// SmoothLife parameters
#define SMOOTHLIFE_B1 0.254
#define SMOOTHLIFE_B2 0.312
#define SMOOTHLIFE_D1 0.340
#define SMOOTHLIFE_D2 0.518
#define SMOOTHLIFE_N 0.028
// Note that these parameters from duckythescientist's port are fixed:
//  sigmode = 2
//  sigtype = 1
//  mixtype = 0
//  timestep_mode = 2

extern "C" __global__ void CreateAntialiasedCircle(uint len_x, uint len_y,
                                                   double* buf, double radius) {
  uint global_x = blockIdx.x * blockDim.x + threadIdx.x;
  uint global_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_x < len_x && global_y < len_y) {
    double center_x = static_cast<double>(len_x) / 2.0;
    double center_y = static_cast<double>(len_y) / 2.0;

    double dst_x = static_cast<double>(global_x) - center_x;
    double dst_y = static_cast<double>(global_y) - center_y;
    double dst = sqrt(dst_x * dst_x + dst_y * dst_y);

    double log_res = log2(static_cast<double>(len_x < len_y ? len_x : len_y));

    double val = 1.0 / (1.0 + exp(log_res * (dst - radius)));

    uint rolled_global_x = (global_x + len_x / 2) % len_x;
    uint rolled_global_y = (global_y + len_y / 2) % len_y;
    uint rolled_flat_idx = rolled_global_y * len_x + rolled_global_x;
    buf[rolled_flat_idx] = val;
  }
}

extern "C" __global__ void SubtractD(uint len_x, uint len_y, double* buf1,
                                     double* buf2) {
  uint global_x = blockIdx.x * blockDim.x + threadIdx.x;
  uint global_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_x < len_x && global_y < len_y) {
    uint flat_idx = global_y * len_x + global_x;
    buf1[flat_idx] -= buf2[flat_idx];
  }
}

extern "C" __global__ void DivideScalarD(uint len_x, uint len_y, double* buf,
                                         double val) {
  uint global_x = blockIdx.x * blockDim.x + threadIdx.x;
  uint global_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_x < len_x && global_y < len_y) {
    uint flat_idx = global_y * len_x + global_x;
    buf[flat_idx] /= val;
  }
}

extern "C" __global__ void MultiplyZ(uint len_x, uint len_y, double2* buf1,
                                     double2* buf2) {
  uint global_x = blockIdx.x * blockDim.x + threadIdx.x;
  uint global_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_x < len_x && global_y < len_y) {
    uint flat_idx = global_y * len_x + global_x;

    double a_real = buf1[flat_idx].x;
    double a_complex = buf1[flat_idx].y;
    double b_real = buf2[flat_idx].x;
    double b_complex = buf2[flat_idx].y;
    double res_real = a_real * b_real - a_complex * b_complex;
    double res_complex = a_real * b_complex + a_complex * b_real;

    buf1[flat_idx].x = res_real;
    buf1[flat_idx].y = res_complex;
  }
}

__device__ inline double Clip(double a, double min, double max) {
  if (a > max) {
    return max;
  } else if (a < min) {
    return min;
  }
  return a;
}

__device__ inline double LinearizedThreshold(double x, double x0,
                                             double alpha) {
  return Clip((x - x0) / alpha + 0.5, 0.0, 1.0);
}

__device__ inline double LinearizedInterval(double x, double a, double b,
                                            double alpha) {
  double res1 = LinearizedThreshold(x, a, alpha);
  double res2 = 1.0 - LinearizedThreshold(x, b, alpha);
  return res1 * res2;
}

extern "C" __global__ void S(uint len_x, uint len_y, double dt, double* world,
                             double* _M, double* _N) {
  uint global_x = blockIdx.x * blockDim.x + threadIdx.x;
  uint global_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_x < len_x && global_y < len_y) {
    uint flat_idx = global_y * len_x + global_x;

    double n = _N[flat_idx];
    double m = _M[flat_idx];
    double f = world[flat_idx];
    double b_thresh =
        LinearizedInterval(n, SMOOTHLIFE_B1, SMOOTHLIFE_B2, SMOOTHLIFE_N);
    double d_thresh =
        LinearizedInterval(n, SMOOTHLIFE_D1, SMOOTHLIFE_D2, SMOOTHLIFE_N);
    double transition = m > 0.5 ? d_thresh : b_thresh;
    double next = f + dt * (transition - f);
    world[flat_idx] = Clip(next, 0.0, 1.0);
  }
}

extern "C" __global__ void Paint(uint len_x, uint len_y, double* buf,
                                 uint32_t* argb) {
  uint global_x = blockIdx.x * blockDim.x + threadIdx.x;
  uint global_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_x < len_x && global_y < len_y) {
    uint flat_idx = global_y * len_x + global_x;
    double val = Clip(buf[flat_idx], 0.0, 1.0);
    uint8_t col = static_cast<uint8_t>(255.0 * val);
    argb[flat_idx] = (col << 16) | (col << 8) | col;
  }
}
