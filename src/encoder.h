#ifndef SMOOTHLIFE_ENCODER_H_
#define SMOOTHLIFE_ENCODER_H_

#include <NVENC/nvEncodeAPI.h>
#include <cuda.h>

#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <queue>

struct EncoderConfig {
  GUID codec;
  GUID preset;
  NV_ENC_TUNING_INFO tuning_info;
};

using DataCallback = std::function<void(uint8_t*, size_t)>;

class Encoder {
 public:
  Encoder(DataCallback callback, int len_x, int len_y, const EncoderConfig& cfg,
          CUcontext cu_ctx);
  Encoder(Encoder&) = delete;
  Encoder& operator=(const Encoder&) = delete;
  Encoder(Encoder&&) = delete;
  Encoder& operator=(Encoder&&) = delete;
  ~Encoder();
  void EncodeFrame(CUdeviceptr frame);
  void EncodeEnd();

 private:
  class NvEncBuffers {
   public:
    NvEncBuffers(NV_ENCODE_API_FUNCTION_LIST* nvenc, void* encoder, int len_x,
                 int len_y);
    NvEncBuffers(NvEncBuffers&) = delete;
    NvEncBuffers& operator=(NvEncBuffers&) = delete;
    NvEncBuffers(NvEncBuffers&&) = delete;
    NvEncBuffers& operator=(NvEncBuffers&&) = delete;
    ~NvEncBuffers();

    void MapInput(NV_ENCODE_API_FUNCTION_LIST* nvenc, void* encoder);
    void UnmapInput(NV_ENCODE_API_FUNCTION_LIST* nvenc, void* encoder);
    void Destroy(NV_ENCODE_API_FUNCTION_LIST* nvenc, void* encoder);

    NV_ENC_INPUT_PTR input_buf() const { return input_buf_; }
    NV_ENC_OUTPUT_PTR output_buf() const { return output_buf_; }
    CUdeviceptr frame_buf() const { return frame_buf_; }

   private:
    NV_ENC_INPUT_PTR input_buf_{};
    NV_ENC_OUTPUT_PTR output_buf_;
    NV_ENC_REGISTERED_PTR frame_resource_;
    CUdeviceptr frame_buf_;
  };
  void InitEncoder(const EncoderConfig& cfg, CUcontext cu_ctx);
  std::unique_ptr<NvEncBuffers> GetFreshBuffers();
  void RecvEncodedData();

  int len_x_;
  int len_y_;

  NV_ENCODE_API_FUNCTION_LIST nvenc_{};
  void* encoder_;
  size_t min_frames_in_use_;
  // Buffers not in use by NVENC
  std::queue<std::unique_ptr<NvEncBuffers>> free_buffers_;
  // Buffers in use by NVENC
  std::queue<std::unique_ptr<NvEncBuffers>> in_use_buffers_;

  DataCallback callback_;
};

#endif  // SMOOTHLIFE_ENCODER_H_
