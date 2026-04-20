#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "autotune.h"
#include "base/add.h"
#include "nvidia/add/kernel.h"
#include "nvidia/runtime_.h"
#include "operator.h"
#include "tensor.h"

int main(int argc, char** argv) {
  using namespace infini::ops;

  // 默认把缓存文件放在 scripts/ 下
  std::filesystem::path default_cache_path =
      std::filesystem::absolute(argv[0]).parent_path() /
      "autotune_warmup_smoke_add_nvidia.cache";

  // 允许通过命令行传入缓存文件路径；不传则使用同目录默认路径
  const std::string cache_path =
      argc > 1 ? std::string(argv[1]) : default_cache_path.string();

  // 每次运行先删除旧缓存，避免历史结果干扰本次流程验证
  std::remove(cache_path.c_str());

  // 这里选一个中等规模 shape，既能覆盖 kernel 执行，又不至于太慢
  std::vector<Tensor::Size> shape{1, 4096};
  const auto n = static_cast<size_t>(shape[0] * shape[1]);

  std::vector<float> h_input(n, 1.0f);
  std::vector<float> h_other(n, 2.0f);
  std::vector<float> h_out(n, 0.0f);

  void* d_input = nullptr;
  void* d_other = nullptr;
  void* d_out = nullptr;

  Runtime<Device::Type::kNvidia>::Malloc(&d_input, n * sizeof(float));
  Runtime<Device::Type::kNvidia>::Malloc(&d_other, n * sizeof(float));
  Runtime<Device::Type::kNvidia>::Malloc(&d_out, n * sizeof(float));

  Runtime<Device::Type::kNvidia>::Memcpy(
      d_input, h_input.data(), n * sizeof(float),
      Runtime<Device::Type::kNvidia>::MemcpyHostToDevice);
  Runtime<Device::Type::kNvidia>::Memcpy(
      d_other, h_other.data(), n * sizeof(float),
      Runtime<Device::Type::kNvidia>::MemcpyHostToDevice);

  Device dev{Device::Type::kNvidia};
  Tensor input{d_input, shape, DataType::kFloat32, dev};
  Tensor other{d_other, shape, DataType::kFloat32, dev};
  Tensor out{d_out, shape, DataType::kFloat32, dev};

  Handle handle;
  Config config;
  config.set_implementation_index(0);

  // 手动构造 query，后续用于读取“该 key 是否已有调优结果”
  std::string op_name = typeid(Add).name();
  auto key = detail::CacheKey::Build(op_name, Device::Type::kNvidia,
                                     config.implementation_index(), input,
                                     other, out);
  autotune::Query query;
  query.key_hash = key.hash;
  query.op_name = op_name;
  query.device_type = Device::Type::kNvidia;
  query.implementation_index = 0;
  query.param_name = "cuda_add_block_size";

  // 阶段 1（WarmupRecord）只记录 key，不应出现调优参数
  autotune::EnableWarmupRecord(cache_path);
  Add::call(handle, config, input, other, out);
  auto warmup_value = autotune::GetTunedInt(query);
  std::cout << "warmup_has_tuned_value="
            << (warmup_value.has_value() ? "yes" : "no") << "\n";

  // 阶段 2（TuneAndUse）应执行调优并得到 block_size
  autotune::EnableTuneAndUse(cache_path);
  Add::call(handle, config, input, other, out);
  auto tuned_value = autotune::GetTunedInt(query);
  std::cout << "tune_has_tuned_value="
            << (tuned_value.has_value() ? "yes" : "no");
  if (tuned_value.has_value()) {
    std::cout << " block_size=" << *tuned_value;
  }
  std::cout << "\n";

  // 显式 flush 后检查缓存文件存在，验证“调优结果已缓存”
  autotune::Flush();
  std::ifstream in(cache_path);
  std::cout << "cache_file_exists=" << (in.good() ? "yes" : "no") << "\n";

  // 阶段 3（UseCacheOnly）应命中缓存，不再重新调优
  autotune::Disable();
  autotune::EnableUseCacheOnly(cache_path);
  Add::call(handle, config, input, other, out);
  auto cache_hit_value = autotune::GetTunedInt(query);
  std::cout << "cache_hit_has_tuned_value="
            << (cache_hit_value.has_value() ? "yes" : "no");
  if (cache_hit_value.has_value()) {
    std::cout << " block_size=" << *cache_hit_value;
  }
  std::cout << "\n";

  // 最终做数值正确性校验，确保自动调优流程不影响结果
  Runtime<Device::Type::kNvidia>::Memcpy(
      h_out.data(), d_out, n * sizeof(float),
      Runtime<Device::Type::kNvidia>::MemcpyDeviceToHost);
  float max_err = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    max_err = std::max(max_err, std::abs(h_out[i] - 3.0f));
  }
  std::cout << "max_err=" << max_err << "\n";

  Runtime<Device::Type::kNvidia>::Free(d_out);
  Runtime<Device::Type::kNvidia>::Free(d_other);
  Runtime<Device::Type::kNvidia>::Free(d_input);

  // 返回码用于自动化判断是否通过
  const bool ok =
      !warmup_value.has_value() && tuned_value.has_value() &&
      cache_hit_value.has_value() && max_err == 0.0f;
  return ok ? 0 : 1;
}
