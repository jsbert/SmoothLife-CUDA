#include "encoder.h"

#include <NVENC/nvEncodeAPI.h>
#include <cuda.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "util.h"

Encoder::Encoder(DataCallback callback, int len_x, int len_y,
                 const EncoderConfig& cfg, CUcontext cu_ctx)
    : len_x_(len_x), len_y_(len_y), callback_(std::move(callback)) {
  uint32_t max_version = 0;
  uint32_t header_version =
      (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;
  NVENC_SUCCEED_NOENC(NvEncodeAPIGetMaxSupportedVersion(&max_version));
  if (header_version > max_version) {
    Panic("Driver does not support the NVENC version we were compiled with.");
  }

  nvenc_.version = NV_ENCODE_API_FUNCTION_LIST_VER;
  NVENC_SUCCEED_NOENC(NvEncodeAPICreateInstance(&nvenc_));

  NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS open_args = {};
  open_args.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
  open_args.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
  open_args.device = cu_ctx;
  open_args.apiVersion = NVENCAPI_VERSION;
  NVENC_SUCCEED_NOENC(nvenc_.nvEncOpenEncodeSessionEx(&open_args, &encoder_));

  const GUID encode_guid = cfg.codec;
  const GUID preset_guid = cfg.preset;
  const NV_ENC_TUNING_INFO tuning_info = cfg.tuning_info;

  NV_ENC_PRESET_CONFIG preset_config = {};
  preset_config.version = NV_ENC_PRESET_CONFIG_VER;
  preset_config.presetCfg.version = NV_ENC_CONFIG_VER;
  NVENC_SUCCEED(
      &nvenc_, encoder_,
      nvenc_.nvEncGetEncodePresetConfigEx(encoder_, encode_guid, preset_guid,
                                          tuning_info, &preset_config));

  min_frames_in_use_ = preset_config.presetCfg.rcParams.lookaheadDepth +
                       preset_config.presetCfg.frameIntervalP + 5;

  NV_ENC_INITIALIZE_PARAMS init_args = {};
  init_args.version = NV_ENC_INITIALIZE_PARAMS_VER;
  init_args.encodeGUID = encode_guid;
  init_args.presetGUID = preset_guid;
  init_args.tuningInfo = tuning_info;
  init_args.encodeWidth = len_x_;
  init_args.encodeHeight = len_y_;
  init_args.frameRateNum = 30;
  init_args.frameRateDen = 1;
  init_args.enableEncodeAsync = 0;
  // "If the client wants to send the input buffers in display order, it must
  // set enablePTD = 1"
  init_args.enablePTD = 1;
  init_args.encodeConfig = &preset_config.presetCfg;
  NVENC_SUCCEED(&nvenc_, encoder_,
                nvenc_.nvEncInitializeEncoder(encoder_, &init_args));
}

Encoder::~Encoder() {
  assert(in_use_buffers_.empty());

  while (!free_buffers_.empty()) {
    free_buffers_.front()->Destroy(&nvenc_, encoder_);
    free_buffers_.pop();
  }

  NVENC_SUCCEED_NOENC(nvenc_.nvEncDestroyEncoder(encoder_));
}

std::unique_ptr<Encoder::NvEncBuffers> Encoder::GetFreshBuffers() {
  if (!free_buffers_.empty()) {
    std::unique_ptr<NvEncBuffers> ret = std::move(free_buffers_.front());
    free_buffers_.pop();
    return ret;
  }
  return std::make_unique<NvEncBuffers>(&nvenc_, encoder_, len_x_, len_y_);
}

void Encoder::RecvEncodedData() {
  assert(!in_use_buffers_.empty());
  std::unique_ptr<NvEncBuffers> buffers = std::move(in_use_buffers_.front());
  in_use_buffers_.pop();

  NV_ENC_LOCK_BITSTREAM lock_bitstream = {};
  lock_bitstream.version = NV_ENC_LOCK_BITSTREAM_VER;
  lock_bitstream.outputBitstream = buffers->output_buf();
  lock_bitstream.doNotWait = 0;
  NVENC_SUCCEED(&nvenc_, encoder_,
                nvenc_.nvEncLockBitstream(encoder_, &lock_bitstream));

  uint8_t* data = static_cast<uint8_t*>(lock_bitstream.bitstreamBufferPtr);
  size_t size = lock_bitstream.bitstreamSizeInBytes;
  callback_(data, size);

  NVENC_SUCCEED(&nvenc_, encoder_,
                nvenc_.nvEncUnlockBitstream(encoder_, buffers->output_buf()));

  buffers->UnmapInput(&nvenc_, encoder_);

  free_buffers_.push(std::move(buffers));
}

void Encoder::EncodeFrame(CUdeviceptr frame) {
  std::unique_ptr<NvEncBuffers> buffers = GetFreshBuffers();

  CU_SUCCEED(cuMemcpyDtoD(buffers->frame_buf(), frame,
                          len_x_ * len_y_ * sizeof(uint32_t)));

  buffers->MapInput(&nvenc_, encoder_);

  NV_ENC_PIC_PARAMS pic_args = {};
  pic_args.version = NV_ENC_PIC_PARAMS_VER;
  pic_args.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
  pic_args.inputBuffer = buffers->input_buf();
  pic_args.bufferFmt = NV_ENC_BUFFER_FORMAT_ARGB;
  pic_args.inputWidth = len_x_;
  pic_args.inputHeight = len_y_;
  pic_args.outputBitstream = buffers->output_buf();
  NVENCSTATUS status = nvenc_.nvEncEncodePicture(encoder_, &pic_args);

  in_use_buffers_.push(std::move(buffers));

  if (status != NV_ENC_ERR_NEED_MORE_INPUT) {
    NVENC_SUCCEED(&nvenc_, encoder_, status);
    while (in_use_buffers_.size() > min_frames_in_use_) {
      RecvEncodedData();
    }
  }
}

void Encoder::EncodeEnd() {
  // Send EOS notification
  NV_ENC_PIC_PARAMS pic_args = {};
  pic_args.version = NV_ENC_PIC_PARAMS_VER;
  pic_args.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
  NVENC_SUCCEED(&nvenc_, encoder_,
                nvenc_.nvEncEncodePicture(encoder_, &pic_args));

  // Receive any remaining encoded data
  while (!in_use_buffers_.empty()) {
    RecvEncodedData();
  }
}

Encoder::NvEncBuffers::NvEncBuffers(NV_ENCODE_API_FUNCTION_LIST* nvenc,
                                    void* encoder, int len_x, int len_y) {
  NV_ENC_CREATE_BITSTREAM_BUFFER create_bitstream_buf = {};
  create_bitstream_buf.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
  NVENC_SUCCEED(
      nvenc, encoder,
      nvenc->nvEncCreateBitstreamBuffer(encoder, &create_bitstream_buf));
  output_buf_ = create_bitstream_buf.bitstreamBuffer;

  CU_SUCCEED(cuMemAlloc(&frame_buf_, len_x * len_y * sizeof(uint32_t)));

  NV_ENC_REGISTER_RESOURCE register_res = {};
  register_res.version = NV_ENC_REGISTER_RESOURCE_VER;
  register_res.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
  register_res.resourceToRegister = reinterpret_cast<void*>(frame_buf_);
  register_res.width = len_x;
  register_res.height = len_y;
  register_res.pitch = len_x * sizeof(uint32_t);
  register_res.bufferFormat = NV_ENC_BUFFER_FORMAT_ARGB;
  NVENC_SUCCEED(nvenc, encoder,
                nvenc->nvEncRegisterResource(encoder, &register_res));
  frame_resource_ = register_res.registeredResource;
}

Encoder::NvEncBuffers::~NvEncBuffers() {
  assert(input_buf_ == nullptr);
  assert(output_buf_ == nullptr);
  assert(frame_resource_ == nullptr);
  assert(frame_buf_ == 0);
}

void Encoder::NvEncBuffers::MapInput(NV_ENCODE_API_FUNCTION_LIST* nvenc,
                                     void* encoder) {
  assert(input_buf_ == nullptr);
  NV_ENC_MAP_INPUT_RESOURCE map_input_res = {};
  map_input_res.version = NV_ENC_MAP_INPUT_RESOURCE_VER;
  map_input_res.registeredResource = frame_resource_;
  NVENC_SUCCEED(nvenc, encoder,
                nvenc->nvEncMapInputResource(encoder, &map_input_res));
  input_buf_ = map_input_res.mappedResource;
}

void Encoder::NvEncBuffers::UnmapInput(NV_ENCODE_API_FUNCTION_LIST* nvenc,
                                       void* encoder) {
  assert(input_buf_ != nullptr);
  NVENC_SUCCEED(nvenc, encoder,
                nvenc->nvEncUnmapInputResource(encoder, input_buf_));
  input_buf_ = nullptr;
}

void Encoder::NvEncBuffers::Destroy(NV_ENCODE_API_FUNCTION_LIST* nvenc,
                                    void* encoder) {
  if (input_buf_ != nullptr) {
    UnmapInput(nvenc, encoder);
  }
  if (output_buf_ != nullptr) {
    NVENC_SUCCEED(nvenc, encoder,
                  nvenc->nvEncDestroyBitstreamBuffer(encoder, output_buf_));
    output_buf_ = nullptr;
  }
  if (frame_resource_ != nullptr) {
    NVENC_SUCCEED(nvenc, encoder,
                  nvenc->nvEncUnregisterResource(encoder, frame_resource_));
    frame_resource_ = nullptr;
  }
  if (frame_buf_ != 0) {
    CU_SUCCEED(cuMemFree(frame_buf_));
    frame_buf_ = 0;
  }
}
