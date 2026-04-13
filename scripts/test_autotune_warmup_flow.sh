#!/usr/bin/env sh
set -eu

# 仓库根目录（POSIX 写法，兼容 `sh`）
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

# 源文件与可执行文件保存在 scripts/ 目录
SRC="${ROOT_DIR}/scripts/autotune_warmup_smoke_nvidia.cu"
BIN="${ROOT_DIR}/scripts/autotune_warmup_smoke_nvidia"

# 步骤 1 - 构建 InfiniOps（WITH_NVIDIA=ON）
cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/build_nvidia" \
  -DWITH_CPU=ON \
  -DWITH_NVIDIA=ON \
  -DGENERATE_PYTHON_BINDINGS=OFF
cmake --build "${ROOT_DIR}/build_nvidia" -j"$(nproc)"

# 步骤 2 - 编译最小验证程序，输出到 scripts/autotune_warmup_smoke_nvidia
nvcc -std=c++17 -O2 "${SRC}" -x cu \
  -I"${ROOT_DIR}/src" \
  -I"/usr/local/cuda-12.8/include" \
  -DWITH_NVIDIA=1 \
  -L"${ROOT_DIR}/build_nvidia/src" -linfiniops \
  -Xlinker -rpath,"${ROOT_DIR}/build_nvidia/src" \
  -L"/usr/local/cuda-12.8/lib64" -lcudart \
  -o "${BIN}"

# 步骤 3 - 运行验证程序
"${BIN}"
