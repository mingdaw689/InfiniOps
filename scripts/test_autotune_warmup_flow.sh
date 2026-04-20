#!/usr/bin/env sh
set -u

# 仓库根目录（POSIX 写法，兼容 `sh`）
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

# 源文件与可执行文件保存在 scripts/ 目录
ADD_SRC="${ROOT_DIR}/scripts/autotune_warmup_smoke_add_nvidia.cu"
ADD_BIN="${ROOT_DIR}/scripts/autotune_warmup_smoke_add_nvidia"
CAUSAL_SRC="${ROOT_DIR}/scripts/autotune_warmup_smoke_causal_softmax_nvidia.cu"
CAUSAL_BIN="${ROOT_DIR}/scripts/autotune_warmup_smoke_causal_softmax_nvidia"
SWIGLU_SRC="${ROOT_DIR}/scripts/autotune_warmup_smoke_swiglu_nvidia.cu"
SWIGLU_BIN="${ROOT_DIR}/scripts/autotune_warmup_smoke_swiglu_nvidia"
RMSNORM_SRC="${ROOT_DIR}/scripts/autotune_warmup_smoke_rms_norm_nvidia.cu"
RMSNORM_BIN="${ROOT_DIR}/scripts/autotune_warmup_smoke_rms_norm_nvidia"

# 步骤 1 - 构建 InfiniOps（WITH_NVIDIA=ON）
echo "[STEP 1] build_nvidia"
cmake -S "${ROOT_DIR}" -B "${ROOT_DIR}/build_nvidia" \
  -DWITH_CPU=ON \
  -DWITH_NVIDIA=ON \
  -DGENERATE_PYTHON_BINDINGS=OFF
cmake --build "${ROOT_DIR}/build_nvidia" -j"$(nproc)"

# 步骤 2 - 编译并运行 Add 最小验证程序
echo "[STEP 2] compile add smoke"
nvcc -std=c++17 -O2 "${ADD_SRC}" -x cu \
  -I"${ROOT_DIR}/src" \
  -I"/usr/local/cuda-12.8/include" \
  -DWITH_NVIDIA=1 \
  -L"${ROOT_DIR}/build_nvidia/src" -linfiniops \
  -Xlinker -rpath,"${ROOT_DIR}/build_nvidia/src" \
  -L"/usr/local/cuda-12.8/lib64" -lcudart \
  -o "${ADD_BIN}" || {
  echo "[ERROR] compile add smoke failed"
  exit 1
}

# 步骤 3 - 编译并运行 CausalSoftmax 最小验证程序
echo "[STEP 3] compile causal_softmax smoke"
nvcc -std=c++17 -O2 "${CAUSAL_SRC}" -x cu \
  -I"${ROOT_DIR}/src" \
  -I"/usr/local/cuda-12.8/include" \
  -DWITH_NVIDIA=1 \
  -L"${ROOT_DIR}/build_nvidia/src" -linfiniops \
  -Xlinker -rpath,"${ROOT_DIR}/build_nvidia/src" \
  -L"/usr/local/cuda-12.8/lib64" -lcudart \
  -o "${CAUSAL_BIN}" || {
  echo "[ERROR] compile causal_softmax smoke failed"
  exit 1
}

# 步骤 4 - 编译 Swiglu 最小验证程序
echo "[STEP 4] compile swiglu smoke"
nvcc -std=c++17 -O2 "${SWIGLU_SRC}" -x cu \
  -I"${ROOT_DIR}/src" \
  -I"/usr/local/cuda-12.8/include" \
  -DWITH_NVIDIA=1 \
  -L"${ROOT_DIR}/build_nvidia/src" -linfiniops \
  -Xlinker -rpath,"${ROOT_DIR}/build_nvidia/src" \
  -L"/usr/local/cuda-12.8/lib64" -lcudart \
  -o "${SWIGLU_BIN}" || {
  echo "[ERROR] compile swiglu smoke failed"
  exit 1
}

# 步骤 5 - 编译 RmsNorm 最小验证程序
echo "[STEP 5] compile rms_norm smoke"
nvcc -std=c++17 -O2 "${RMSNORM_SRC}" -x cu \
  -I"${ROOT_DIR}/src" \
  -I"/usr/local/cuda-12.8/include" \
  -DWITH_NVIDIA=1 \
  -L"${ROOT_DIR}/build_nvidia/src" -linfiniops \
  -Xlinker -rpath,"${ROOT_DIR}/build_nvidia/src" \
  -L"/usr/local/cuda-12.8/lib64" -lcudart \
  -o "${RMSNORM_BIN}" || {
  echo "[ERROR] compile rms_norm smoke failed"
  exit 1
}

status=0

echo "[STEP 6] run add smoke"
if ! "${ADD_BIN}"; then
  echo "[ERROR] add smoke failed"
  status=1
fi

echo "[STEP 7] run causal_softmax smoke"
if ! "${CAUSAL_BIN}"; then
  echo "[ERROR] causal_softmax smoke failed"
  status=1
fi

echo "[STEP 8] run swiglu smoke"
if ! "${SWIGLU_BIN}"; then
  echo "[ERROR] swiglu smoke failed"
  status=1
fi

echo "[STEP 9] run rms_norm smoke"
if ! "${RMSNORM_BIN}"; then
  echo "[ERROR] rms_norm smoke failed"
  status=1
fi

exit "${status}"
