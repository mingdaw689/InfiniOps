#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "autotune.h"
#include "base/causal_softmax.h"
#include "nvidia/causal_softmax/kernel.h"
#include "nvidia/runtime_.h"
#include "operator.h"
#include "tensor.h"

namespace {

// 中文注释：计算参考结果（下三角可见的 causal softmax），用于校验内核输出正确性。
std::vector<float> BuildCausalSoftmaxReference(const std::vector<float>& input,
                                               std::size_t seq_len,
                                               std::size_t total_seq_len) {
  std::vector<float> ref(input.size(), 0.0f);

  for (std::size_t row = 0; row < seq_len; ++row) {
    float max_val = -std::numeric_limits<float>::infinity();
    for (std::size_t col = 0; col <= row; ++col) {
      max_val = std::max(max_val, input[row * total_seq_len + col]);
    }

    float sum = 0.0f;
    for (std::size_t col = 0; col <= row; ++col) {
      float e = std::exp(input[row * total_seq_len + col] - max_val);
      ref[row * total_seq_len + col] = e;
      sum += e;
    }
    for (std::size_t col = 0; col <= row; ++col) {
      ref[row * total_seq_len + col] /= sum;
    }
    for (std::size_t col = row + 1; col < total_seq_len; ++col) {
      ref[row * total_seq_len + col] = 0.0f;
    }
  }

  return ref;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace infini::ops;

  // 中文注释：默认把缓存文件放在可执行文件同目录，便于定位和清理。
  std::filesystem::path default_cache_path =
      std::filesystem::absolute(argv[0]).parent_path() /
      "autotune_warmup_smoke_causal_softmax_nvidia.cache";
  const std::string cache_path =
      argc > 1 ? std::string(argv[1]) : default_cache_path.string();

  // 中文注释：每次 smoke 先清理旧缓存，确保本次输出是新流程产生的。
  std::remove(cache_path.c_str());

  // 中文注释：采用 2D 张量 [seq_len, total_seq_len]，满足 causal_softmax 的输入约束。
  constexpr std::size_t seq_len = 64;
  constexpr std::size_t total_seq_len = 64;
  std::vector<Tensor::Size> shape{seq_len, total_seq_len};
  const std::size_t n = seq_len * total_seq_len;

  std::vector<float> h_input(n);
  for (std::size_t i = 0; i < n; ++i) {
    h_input[i] = static_cast<float>((i % 17) - 8) / 8.0f;
  }
  std::vector<float> h_out(n, 0.0f);
  const auto h_ref = BuildCausalSoftmaxReference(h_input, seq_len, total_seq_len);

  void* d_input = nullptr;
  void* d_out = nullptr;
  Runtime<Device::Type::kNvidia>::Malloc(&d_input, n * sizeof(float));
  Runtime<Device::Type::kNvidia>::Malloc(&d_out, n * sizeof(float));
  Runtime<Device::Type::kNvidia>::Memcpy(
      d_input, h_input.data(), n * sizeof(float),
      Runtime<Device::Type::kNvidia>::MemcpyHostToDevice);

  Device dev{Device::Type::kNvidia};
  Tensor input{d_input, shape, DataType::kFloat32, dev};
  Tensor out{d_out, shape, DataType::kFloat32, dev};

  Handle handle;
  Config config;
  config.set_implementation_index(0);

  // 中文注释：构造与框架一致的 query key，用于查询当前 key 的调优结果。
  std::string op_name = typeid(CausalSoftmax).name();
  auto key = detail::CacheKey::Build(op_name, Device::Type::kNvidia,
                                     config.implementation_index(), input, out);
  autotune::Query query;
  query.key_hash = key.hash;
  query.op_name = op_name;
  query.device_type = Device::Type::kNvidia;
  query.implementation_index = 0;
  query.param_name = "cuda_causal_softmax_block_size";

  // 中文注释：阶段 1（WarmupRecord）应只记录 key，不产生已选参数。
  autotune::EnableWarmupRecord(cache_path);
  CausalSoftmax::call(handle, config, input, out);
  auto warmup_value = autotune::GetTunedInt(query);
  std::cout << "warmup_has_tuned_value="
            << (warmup_value.has_value() ? "yes" : "no") << "\n";

  // 中文注释：阶段 2（TuneAndUse）应完成候选比较并写入调优值。
  autotune::EnableTuneAndUse(cache_path);
  CausalSoftmax::call(handle, config, input, out);
  auto tuned_value = autotune::GetTunedInt(query);
  std::cout << "tune_has_tuned_value="
            << (tuned_value.has_value() ? "yes" : "no");
  if (tuned_value.has_value()) {
    std::cout << " block_size=" << *tuned_value;
  }
  std::cout << "\n";

  autotune::Flush();
  std::ifstream in(cache_path);
  std::cout << "cache_file_exists=" << (in.good() ? "yes" : "no") << "\n";

  // 中文注释：阶段 3（UseCacheOnly）应直接命中缓存，不再触发调优。
  autotune::Disable();
  autotune::EnableUseCacheOnly(cache_path);
  CausalSoftmax::call(handle, config, input, out);
  auto cache_hit_value = autotune::GetTunedInt(query);
  std::cout << "cache_hit_has_tuned_value="
            << (cache_hit_value.has_value() ? "yes" : "no");
  if (cache_hit_value.has_value()) {
    std::cout << " block_size=" << *cache_hit_value;
  }
  std::cout << "\n";

  Runtime<Device::Type::kNvidia>::Memcpy(
      h_out.data(), d_out, n * sizeof(float),
      Runtime<Device::Type::kNvidia>::MemcpyDeviceToHost);

  float max_err = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    max_err = std::max(max_err, std::abs(h_out[i] - h_ref[i]));
  }
  std::cout << "max_err=" << max_err << "\n";

  Runtime<Device::Type::kNvidia>::Free(d_out);
  Runtime<Device::Type::kNvidia>::Free(d_input);

  const bool ok = !warmup_value.has_value() && tuned_value.has_value() &&
                  cache_hit_value.has_value() && max_err < 1e-4f;
  return ok ? 0 : 1;
}
