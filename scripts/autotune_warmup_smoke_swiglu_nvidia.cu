#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "autotune.h"
#include "base/swiglu.h"
#include "nvidia/runtime_.h"
#include "nvidia/swiglu/kernel.h"
#include "operator.h"
#include "tensor.h"

namespace {

// 中文注释：使用和测试一致的参考公式 out = input * (gate * sigmoid(gate))。
std::vector<float> BuildSwigluReference(const std::vector<float>& input,
                                        const std::vector<float>& gate) {
  std::vector<float> ref(input.size(), 0.0f);
  for (std::size_t i = 0; i < input.size(); ++i) {
    const float sig = 1.0f / (1.0f + std::exp(-gate[i]));
    ref[i] = input[i] * (gate[i] * sig);
  }
  return ref;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace infini::ops;

  // 中文注释：默认缓存放在可执行文件同目录，便于测试后查看和清理。
  std::filesystem::path default_cache_path =
      std::filesystem::absolute(argv[0]).parent_path() /
      "autotune_warmup_smoke_swiglu_nvidia.cache";
  const std::string cache_path =
      argc > 1 ? std::string(argv[1]) : default_cache_path.string();

  std::remove(cache_path.c_str());

  // 中文注释：使用 2D shape，覆盖常见 MLP 场景。
  std::vector<Tensor::Size> shape{16, 5632};
  const std::size_t n = shape[0] * shape[1];

  std::vector<float> h_input(n);
  std::vector<float> h_gate(n);
  for (std::size_t i = 0; i < n; ++i) {
    h_input[i] = static_cast<float>((i % 29) - 14) / 16.0f;
    h_gate[i] = static_cast<float>((i % 31) - 15) / 16.0f;
  }
  std::vector<float> h_out(n, 0.0f);
  const auto h_ref = BuildSwigluReference(h_input, h_gate);

  void* d_input = nullptr;
  void* d_gate = nullptr;
  void* d_out = nullptr;
  Runtime<Device::Type::kNvidia>::Malloc(&d_input, n * sizeof(float));
  Runtime<Device::Type::kNvidia>::Malloc(&d_gate, n * sizeof(float));
  Runtime<Device::Type::kNvidia>::Malloc(&d_out, n * sizeof(float));

  Runtime<Device::Type::kNvidia>::Memcpy(
      d_input, h_input.data(), n * sizeof(float),
      Runtime<Device::Type::kNvidia>::MemcpyHostToDevice);
  Runtime<Device::Type::kNvidia>::Memcpy(
      d_gate, h_gate.data(), n * sizeof(float),
      Runtime<Device::Type::kNvidia>::MemcpyHostToDevice);

  Device dev{Device::Type::kNvidia};
  Tensor input{d_input, shape, DataType::kFloat32, dev};
  Tensor gate{d_gate, shape, DataType::kFloat32, dev};
  Tensor out{d_out, shape, DataType::kFloat32, dev};

  Handle handle;
  Config config;
  config.set_implementation_index(0);

  // 中文注释：构造与框架一致的 query key，用于读取该 key 的调优结果。
  std::string op_name = typeid(Swiglu).name();
  auto key = detail::CacheKey::Build(op_name, Device::Type::kNvidia,
                                     config.implementation_index(), input, gate,
                                     out);
  autotune::Query query;
  query.key_hash = key.hash;
  query.op_name = op_name;
  query.device_type = Device::Type::kNvidia;
  query.implementation_index = 0;
  query.param_name = "cuda_swiglu_block_size";

  autotune::EnableWarmupRecord(cache_path);
  Swiglu::call(handle, config, input, gate, out);
  auto warmup_value = autotune::GetTunedInt(query);
  std::cout << "warmup_has_tuned_value="
            << (warmup_value.has_value() ? "yes" : "no") << "\n";

  autotune::EnableTuneAndUse(cache_path);
  Swiglu::call(handle, config, input, gate, out);
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

  autotune::Disable();
  autotune::EnableUseCacheOnly(cache_path);
  Swiglu::call(handle, config, input, gate, out);
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
  Runtime<Device::Type::kNvidia>::Free(d_gate);
  Runtime<Device::Type::kNvidia>::Free(d_input);

  const bool ok = !warmup_value.has_value() && tuned_value.has_value() &&
                  cache_hit_value.has_value() && max_err < 1e-5f;
  return ok ? 0 : 1;
}
