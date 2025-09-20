#ifndef SMOOTHLIFE_UTIL_H_
#define SMOOTHLIFE_UTIL_H_

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <format>
#include <print>
#include <source_location>
#include <type_traits>

template <class... Args>
[[noreturn]] static inline void PanicAt(std::source_location loc,
                                        std::format_string<Args...> fmt,
                                        Args&&... args) {
  std::print(stderr, "Panicked at {}:{}: ", loc.file_name(), loc.line());
  std::println(stderr, fmt, std::forward<Args>(args)...);
  exit(-1);
}

template <class... Args>
struct SrcLocAndFmt {
  template <class T>
  consteval SrcLocAndFmt(
      const T& s, std::source_location loc = std::source_location::current())
      : fmt{s}, loc{loc} {}

  std::format_string<Args...> fmt;
  std::source_location loc;
};

template <class... Args>
[[noreturn]] static inline void Panic(
    SrcLocAndFmt<std::type_identity_t<Args>...> fmt, Args&&... args) {
  PanicAt(fmt.loc, fmt.fmt, std::forward<Args>(args)...);
}

template <class... Args>
static inline void Dbg(SrcLocAndFmt<std::type_identity_t<Args>...> fmt,
                       Args&&... args) {
  std::print(stderr, "{}:{}: ", fmt.loc.file_name(), fmt.loc.line());
  std::println(stderr, fmt.fmt, std::forward<Args>(args)...);
}

#define EASSERT(x, f) EAssert(x, f, #x)
static inline void EAssert(
    int cond, const char* fn, const char* what,
    std::source_location loc = std::source_location::current()) {
  if (!cond) {
    PanicAt(loc, "Assertion ({}) failed after call to {}: {}", what, fn,
            strerror(errno));
  }
}

#ifdef CUDA_VERSION
#define CU_SUCCEED(x) CudaSucceed(x, #x)
static inline void CudaSucceed(
    CUresult res, const char* call,
    std::source_location loc = std::source_location::current()) {
  if (res != CUDA_SUCCESS) {
    const char* s = nullptr;
    if (cuGetErrorString(res, &s) != CUDA_SUCCESS || s == nullptr) {
      s = "Unknown";
    }
    PanicAt(loc, "CUDA call\n  {}\nfailed with error code {}: {}\n", call,
            (int)res, s);
  }
}
#endif

#ifdef CUBLASAPI
#define CUBLAS_SUCCEED(x) CublasSucceed(x, #x)
static inline void CublasSucceed(
    cublasStatus_t res, const char* call,
    std::source_location loc = std::source_location::current()) {
  if (res != CUBLAS_STATUS_SUCCESS) {
    const char* s = cublasGetStatusString(res);
    if (s == nullptr) {
      s = "Unknown";
    }
    PanicAt(loc, "cuBLAS call\n  {}\nfailed with error code {}: {}\n", call,
            (int)res, s);
  }
}
#endif

#ifdef CUFFTAPI
#define CUFFT_SUCCEED(x) CufftSucceed(x, #x)
static inline void CufftSucceed(
    cufftResult res, const char* call,
    std::source_location loc = std::source_location::current()) {
  if (res != CUFFT_SUCCESS) {
    PanicAt(loc, "cuFFT call\n  {}\nfailed with error code {}\n", call,
            (int)res);
  }
}
#endif

#ifdef NVENCAPI
#define NVENC_SUCCEED(nvenc, enc, x) NvEncSucceed(nvenc, enc, x, #x)
static inline void NvEncSucceed(
    NV_ENCODE_API_FUNCTION_LIST* nvenc, void* encoder, NVENCSTATUS res,
    const char* call,
    std::source_location loc = std::source_location::current()) {
  if (res != NV_ENC_SUCCESS) {
    const char* s = nullptr;
    if (encoder != nullptr && nvenc->nvEncGetLastErrorString != nullptr) {
      s = nvenc->nvEncGetLastErrorString(encoder);
    }
    if (s == nullptr) {
      s = "";
    }
    PanicAt(loc, "NVENC call\n  {}\nfailed with error code {}: {}\n", call,
            (int)res, s);
  }
}
#define NVENC_SUCCEED_NOENC(x) NvEncSucceed(nullptr, nullptr, x, #x)
#endif

#endif  // SMOOTHLIFE_UTIL_H_
