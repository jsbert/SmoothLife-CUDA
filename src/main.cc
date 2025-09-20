#include <NVENC/nvEncodeAPI.h>
#include <cuda.h>
#include <unistd.h>

#include <argparse/argparse.hpp>
#include <array>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <optional>
#include <print>
#include <string>
#include <string_view>
#include <utility>

#include "encoder.h"
#include "smoothlife.h"
#include "util.h"

namespace {
volatile sig_atomic_t sigint_status;

struct Config {
  int device_idx;
  int len_x;
  int len_y;
  int num_frames;
  double smoothlife_dt;
  bool raw;
  EncoderConfig encoder_cfg;
  std::string output_file;
};

class Writer {
 public:
  virtual ~Writer() = default;
  virtual void WriteFrame(CUdeviceptr frame) = 0;
  virtual void WriteEnd() = 0;

 protected:
  Writer() {}
};

class RawWriter : public Writer {
 public:
  RawWriter(FILE* f, int len_x, int len_y)
      : f_(f),
        len_x_(len_x),
        len_y_(len_y),
        host_frame_(new uint32_t[len_x * len_y]) {}

  virtual void WriteFrame(CUdeviceptr frame) override {
    size_t size = len_x_ * len_y_ * sizeof(uint32_t);
    CU_SUCCEED(cuMemcpyDtoH(host_frame_.get(), frame, size));
    size_t num_written = fwrite(host_frame_.get(), 1, size, f_);
    EASSERT(num_written == size, "fwrite");
  }

  virtual void WriteEnd() override {}

 private:
  FILE* f_;
  int len_x_;
  int len_y_;
  std::unique_ptr<uint32_t[]> host_frame_;
};

class EncodedWriter : public Writer {
 public:
  EncodedWriter(FILE* f, int len_x, int len_y, const EncoderConfig& cfg,
                CUcontext cu_ctx)
      : f_(f),
        encoder_([this](uint8_t* data, size_t size) { WriteData(data, size); },
                 len_x, len_y, cfg, cu_ctx) {}

  virtual void WriteFrame(CUdeviceptr frame) override {
    encoder_.EncodeFrame(frame);
  }

  virtual void WriteEnd() override { encoder_.EncodeEnd(); }

 private:
  void WriteData(uint8_t* data, size_t size) {
    size_t num_written = fwrite(data, 1, size, f_);
    EASSERT(num_written == size, "fwrite");
  }
  FILE* f_;
  Encoder encoder_;
};

void Run(const Config& cfg, CUcontext cu_ctx) {
  SmoothLife smoothlife(cfg.len_x, cfg.len_y, cfg.smoothlife_dt);
  std::unique_ptr<Writer> writer;
  FILE* f;

  if (cfg.output_file == "-") {
    f = stdout;
  } else {
    f = fopen(cfg.output_file.c_str(), "w");
    EASSERT(f != nullptr, "fopen");
  }

  if (cfg.raw) {
    writer = std::make_unique<RawWriter>(f, cfg.len_x, cfg.len_y);
  } else {
    writer = std::make_unique<EncodedWriter>(f, cfg.len_x, cfg.len_y,
                                             cfg.encoder_cfg, cu_ctx);
  }

  writer->WriteFrame(smoothlife.frame());

  for (int i = 1; i < cfg.num_frames && !sigint_status; ++i) {
    smoothlife.Step();
    writer->WriteFrame(smoothlife.frame());
  }

  writer->WriteEnd();
  if (f != stdout) {
    fclose(f);
  }
}

template <class T, size_t N>
std::optional<T> ParseArg(
    const std::array<std::pair<std::string_view, T>, N>& possible_values,
    std::string_view arg) {
  for (const auto& [s, val] : possible_values) {
    if (s == arg) {
      return val;
    }
  }
  return {};
}

std::optional<Config> ParseConfig(int argc, char** argv) {
  argparse::ArgumentParser program("smoothlife", "0.0.0");
  program.add_argument("-o", "--output")
      .help("File to write output to")
      .metavar("FILE")
      .default_value("out")
      .nargs(1);
  program.add_argument("-d", "--device")
      .help("Index of GPU to use (run nvidia-smi to find this)")
      .metavar("IDX")
      .default_value(0)
      .nargs(1)
      .scan<'i', int>();
  program.add_argument("-w", "--width")
      .help("Width of the world in pixels")
      .metavar("NUM")
      .default_value(1024)
      .nargs(1)
      .scan<'i', int>();
  program.add_argument("-h", "--height")
      .help("Height of the world in pixels")
      .metavar("NUM")
      .default_value(1024)
      .nargs(1)
      .scan<'i', int>();
  program.add_argument("-n", "--num-frames")
      .help("Number of frames to generate")
      .metavar("NUM")
      .default_value(30)
      .nargs(1)
      .scan<'i', int>();
  program.add_argument("--smoothlife-dt")
      .help("SmoothLife parameter (valid values are between 0.0 and 1.0)")
      .metavar("DT")
      .default_value(0.2)
      .nargs(1)
      .scan<'g', double>();
  program.add_argument("-r", "--raw")
      .help(
          "Output raw frames instead of encoding "
          "(NB: produces huge output files)")
      .flag();
  program.add_argument("--encoder-codec")
      .help("Codec to use for encoding (valid values: h264, hevc, av1)")
      .metavar("CODEC")
      .default_value("h264")
      .nargs(1);
  program.add_argument("--encoder-preset")
      .help(
          "Preset to use for encoding "
          "(valid values: p1, p2, p3, p4, p5, p6, p7)")
      .metavar("PRESET")
      .default_value("p7")
      .nargs(1);
  program.add_argument("--encoder-tuning")
      .help(
          "Tuning info to use for encoding "
          "(valid values: hq lowlatency ultralowlatency lossless uhq)")
      .metavar("TUNING")
      .default_value("hq")
      .nargs(1);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::println(stderr, "{}", err.what());
    std::print(stderr, "{}", program.help().str());
    return {};
  }

  constexpr int kMaxDim = 1 << 13;
  constexpr int kMinDim = 64;
  Config cfg;
  cfg.output_file = program.get<std::string>("--output");
  if (cfg.output_file == "-" && isatty(fileno(stdout))) {
    std::println(stderr, "Refusing to write output to tty");
    return {};
  }

  cfg.device_idx = program.get<int>("--device");
  cfg.len_x = program.get<int>("--width");
  if (cfg.len_x < kMinDim || cfg.len_x > kMaxDim) {
    std::println(stderr,
                 "Width of {} is invalid. "
                 "Width must be must be at least {} and at most {}",
                 cfg.len_x, kMinDim, kMaxDim);
    return {};
  }

  cfg.len_y = program.get<int>("--height");
  if (cfg.len_y < kMinDim || cfg.len_y > kMaxDim) {
    std::println(stderr,
                 "Height of {} is invalid. "
                 "Height must be at least {} and at most {}",
                 cfg.len_y, kMinDim, kMaxDim);
    return {};
  }

  cfg.num_frames = program.get<int>("--num-frames");
  if (cfg.num_frames <= 0) {
    std::println(stderr, "Frame count must be greater than zero");
    return {};
  }

  cfg.smoothlife_dt = program.get<double>("--smoothlife-dt");
  if (cfg.smoothlife_dt < 0.0 || cfg.smoothlife_dt > 1.0) {
    std::println(stderr, "SmoothLife delta time must be between 0.0 and 1.0");
    return {};
  }

  cfg.raw = program["-r"] == true;

  const std::array<std::pair<std::string_view, GUID>, 3> kCodecs = {{
      {"h264", NV_ENC_CODEC_H264_GUID},
      {"hevc", NV_ENC_CODEC_HEVC_GUID},
      {"av1", NV_ENC_CODEC_AV1_GUID},
  }};
  std::string codec_str = program.get<std::string>("--encoder-codec");
  auto codec = ParseArg(kCodecs, codec_str);
  if (!codec) {
    std::println(stderr, "{} is not a valid codec.", codec_str);
    return {};
  }
  cfg.encoder_cfg.codec = *codec;

  const std::array<std::pair<std::string_view, GUID>, 7> kPresets = {{
      {"p1", NV_ENC_PRESET_P1_GUID},
      {"p2", NV_ENC_PRESET_P2_GUID},
      {"p3", NV_ENC_PRESET_P3_GUID},
      {"p4", NV_ENC_PRESET_P4_GUID},
      {"p5", NV_ENC_PRESET_P5_GUID},
      {"p6", NV_ENC_PRESET_P6_GUID},
      {"p7", NV_ENC_PRESET_P7_GUID},
  }};
  std::string preset_str = program.get<std::string>("--encoder-preset");
  auto preset = ParseArg(kPresets, preset_str);
  if (!preset) {
    std::println(stderr, "{} is not a valid preset.", preset_str);
    return {};
  }
  cfg.encoder_cfg.preset = *preset;

  const std::array<std::pair<std::string_view, NV_ENC_TUNING_INFO>, 5>
      kTuningInfos = {{
          {"hq", NV_ENC_TUNING_INFO_HIGH_QUALITY},
          {"lowlatency", NV_ENC_TUNING_INFO_LOW_LATENCY},
          {"ultralowlatency", NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY},
          {"lossless", NV_ENC_TUNING_INFO_LOSSLESS},
          {"uhq", NV_ENC_TUNING_INFO_ULTRA_HIGH_QUALITY},
      }};
  std::string tuning_info_str = program.get<std::string>("--encoder-tuning");
  auto tuning_info = ParseArg(kTuningInfos, tuning_info_str);
  if (!tuning_info) {
    std::println(stderr, "{} is not a valid tuning info.", tuning_info_str);
    return {};
  }
  cfg.encoder_cfg.tuning_info = *tuning_info;

  return cfg;
}

void sigint_handler(int signal) {
  (void)signal;
  sigint_status = 1;
}

}  // namespace

int main(int argc, char** argv) {
  signal(SIGINT, sigint_handler);
  auto cfg = ParseConfig(argc, argv);
  if (!cfg) {
    return 1;
  }

  CU_SUCCEED(cuInit(0));

  int num_devices;
  CU_SUCCEED(cuDeviceGetCount(&num_devices));
  if (cfg->device_idx >= num_devices || cfg->device_idx < 0) {
    std::println(stderr, "{} is not a valid device index", cfg->device_idx);
    return 1;
  }

  CUdevice dev;
  CU_SUCCEED(cuDeviceGet(&dev, cfg->device_idx));

  CUcontext cu_ctx;
#if CUDA_VERSION >= 13000
  CUctxCreateParams ctx_create_args = {};
  CU_SUCCEED(cuCtxCreate(&cu_ctx, &ctx_create_args, 0, dev));
#else
  CU_SUCCEED(cuCtxCreate(&cu_ctx, 0, dev));
#endif

  Run(*cfg, cu_ctx);

  CU_SUCCEED(cuCtxDestroy(cu_ctx));
  return 0;
}
