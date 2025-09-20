# SmoothLife-CUDA

CUDA port of
[duckythescientist's Python implementation](https://github.com/duckythescientist/SmoothLife)
of [SmoothLife](https://arxiv.org/abs/1111.1567).

## Dependencies
* CUDA driver API, cuBLAS, cuFFT
* NVENC
* [argparse](https://github.com/p-ranav/argparse) (downloaded automatically
  during build)

Tested with CUDA versions 12.9 and 13.0.

You will need a recent C++ compiler (a few C++23 and C++26 features are used).

## Building
```
$ mkdir build && cd build
$ cmake ..
$ make -j$(nproc)
```

## Usage
Generate 1000 frames of the simulation, saving them as H.264-encoded data in
`out.h264`:
```
$ ./smoothlife --device 0 --num-frames 1000 --width 1920 --height 1080 --encoder-codec h264 --output out.h264
```
Valid codecs for `--encoder-codec` are: `h264`, `hevc`, or (if your GPU supports it) `av1`.

You can write output to `stdout` by specifying `--output -`.
For example, if you have access to a GPU remotely and want to stream to your
local media player:
```
$ ssh machinewithgpu /path/to/smoothlife --device 0 --num-frames 1000 --width 1920 --height 1080 --encoder-codec hevc --output - | mpv -
```

See `./smoothlife --help` for the full list of command-line options.

