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
#include "base/rms_norm.h"
#include "nvidia/rms_norm/kernel.h"
#include "nvidia/runtime_.h"
#include "operator.h"
#include "tensor.h"

namespace {

// 中文注释：参考实现与 CPU 路径一致：y = x / sqrt(mean(x^2) + eps) * weight。
std::vector<float> BuildRmsNormReference(const std::vector<float>& input,
                                         const std::vector<float>& weight,
                                         std::size_t batch,
                                         std::size_t nhead, std::size_t dim,
                                         float eps) {
  std::vector<float> ref(input.size(), 0.0f);
  const std::size_t rows = batch * nhead;
  for (std::size_t r = 0; r < rows; ++r) {
    const std::size_t base = r * dim;
    float ss = 0.0f;
    for (std::size_t k = 0; k < dim; ++k) {
      ss += input[base + k] * input[base + k];
    }
    const float rms = 1.0f / std::sqrt(ss / static_cast<float>(dim) + eps);
    for (std::size_t k = 0; k < dim; ++k) {
      ref[base + k] = input[base + k] * weight[k] * rms;
    }
  }
  return ref;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace infini::ops;

  std::filesystem::path default_cache_path =
      std::filesystem::absolute(argv[0]).parent_path() /
      "autotune_warmup_smoke_rms_norm_nvidia.cache";
  const std::string cache_path =
      argc > 1 ? std::string(argv[1]) : default_cache_path.string();

  std::remove(cache_path.c_str());

  constexpr std::size_t batch = 2;
  constexpr std::size_t nhead = 4;
  constexpr std::size_t dim = 2048;
  std::vector<Tensor::Size> input_shape{batch, nhead, dim};
  std::vector<Tensor::Size> weight_shape{dim};
  const std::size_t n = batch * nhead * dim;

  std::vector<float> h_input(n);
  std::vector<float> h_weight(dim);
  for (std::size_t i = 0; i < n; ++i) {
    h_input[i] = static_cast<float>((i % 19) - 9) / 16.0f;
  }
  for (std::size_t i = 0; i < dim; ++i) {
    h_weight[i] = 0.5f + static_cast<float>((i % 13) - 6) / 32.0f;
  }
  std::vector<float> h_out(n, 0.0f);

  constexpr float eps = 1e-6f;
  const auto h_ref = BuildRmsNormReference(h_input, h_weight, batch, nhead, dim, eps);

  void* d_input = nullptr;
  void* d_weight = nullptr;
  void* d_out = nullptr;
  Runtime<Device::Type::kNvidia>::Malloc(&d_input, n * sizeof(float));
  Runtime<Device::Type::kNvidia>::Malloc(&d_weight, dim * sizeof(float));
  Runtime<Device::Type::kNvidia>::Malloc(&d_out, n * sizeof(float));

  Runtime<Device::Type::kNvidia>::Memcpy(
      d_input, h_input.data(), n * sizeof(float),
      Runtime<Device::Type::kNvidia>::MemcpyHostToDevice);
  Runtime<Device::Type::kNvidia>::Memcpy(
      d_weight, h_weight.data(), dim * sizeof(float),
      Runtime<Device::Type::kNvidia>::MemcpyHostToDevice);

  Device dev{Device::Type::kNvidia};
  Tensor input{d_input, input_shape, DataType::kFloat32, dev};
  Tensor weight{d_weight, weight_shape, DataType::kFloat32, dev};
  Tensor out{d_out, input_shape, DataType::kFloat32, dev};

  Handle handle;
  Config config;
  config.set_implementation_index(0);

  std::string op_name = typeid(RmsNorm).name();
  auto key = detail::CacheKey::Build(op_name, Device::Type::kNvidia,
                                     config.implementation_index(), input,
                                     weight, eps, out);
  autotune::Query query;
  query.key_hash = key.hash;
  query.op_name = op_name;
  query.device_type = Device::Type::kNvidia;
  query.implementation_index = 0;
  query.param_name = "cuda_rms_norm_block_size";

  autotune::EnableWarmupRecord(cache_path);
  RmsNorm::call(handle, config, input, weight, eps, out);
  auto warmup_value = autotune::GetTunedInt(query);
  std::cout << "warmup_has_tuned_value="
            << (warmup_value.has_value() ? "yes" : "no") << "\n";

  autotune::EnableTuneAndUse(cache_path);
  RmsNorm::call(handle, config, input, weight, eps, out);
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
  RmsNorm::call(handle, config, input, weight, eps, out);
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
  Runtime<Device::Type::kNvidia>::Free(d_weight);
  Runtime<Device::Type::kNvidia>::Free(d_input);

  const bool ok = !warmup_value.has_value() && tuned_value.has_value() &&
                  cache_hit_value.has_value() && max_err < 1e-4f;
  return ok ? 0 : 1;
}
