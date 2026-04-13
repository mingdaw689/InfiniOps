#ifndef INFINI_OPS_OPERATOR_H_
#define INFINI_OPS_OPERATOR_H_

#include <cassert>
#include <algorithm>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "autotune.h"
#include "autotune_policy.h"
#include "config.h"
#include "dispatcher.h"
#include "handle.h"
#include "tensor.h"

#ifdef WITH_NVIDIA
// 仅在 NVIDIA 构建下启用 CUDA Event 计时，避免影响其他后端编译
#include <cuda_runtime.h>
#endif

namespace infini::ops::detail {

struct CacheKey {
  std::size_t hash;

  std::vector<Tensor> tensors;

  std::size_t scalar_hash;

  template <typename... Args>
  static CacheKey Build(const Args&... args) {
    CacheKey key;
    key.hash = 0;
    key.scalar_hash = 0;
    (key.Absorb(args), ...);
    return key;
  }

 private:
  void Absorb(const Tensor& t) {
    HashCombine(hash, t);
    tensors.push_back(t);
  }

  template <typename T>
  void Absorb(const T& v) {
    HashCombine(hash, v);
    HashCombine(scalar_hash, v);
  }
};

template <typename Functor, typename... Args, auto... implementation_indices>
auto DispatchImplementation(std::size_t implementation_index, Functor&& func,
                            std::string_view context_str,
                            List<implementation_indices...>, Args&&... args) {
  return DispatchFunc<std::size_t,
                      static_cast<std::size_t>(implementation_indices)...>(
      implementation_index, std::forward<Functor>(func), context_str,
      std::forward<Args>(args)...);
}

template <auto... values>
std::vector<std::size_t> ListToVector(List<values...>) {
  return {static_cast<std::size_t>(values)...};
}

inline bool IsAllowedCudaBlockSize(int block_size) {
  return block_size == 128 || block_size == 256 || block_size == 512 ||
         block_size == 1024 || block_size == 2048;
}

inline std::string OperatorNameFromTypeid(const std::type_info& info) {
  return info.name();
}

inline bool ExtractFirstTensorDeviceType(Device::Type*) { return false; }

template <typename T, typename... Rest>
bool ExtractFirstTensorDeviceType(Device::Type* out, const T& arg,
                                  const Rest&... rest) {
  if constexpr (std::is_same_v<std::decay_t<T>, Tensor>) {
    *out = arg.device().type();
    return true;
  }
  return ExtractFirstTensorDeviceType(out, rest...);
}

#ifdef WITH_NVIDIA
template <typename OpType, typename ApplyFn, typename... Args>
std::optional<int> TuneNvidiaIntCandidates(const Handle& handle, OpType& op,
                                           const Config& base_config,
                                           const std::vector<int>& candidates,
                                           ApplyFn&& apply_candidate,
                                           Args&... args) {
  if (candidates.empty()) return std::nullopt;

  std::vector<std::pair<int, float>> measurements;
  measurements.reserve(candidates.size());

  auto stream = static_cast<cudaStream_t>(handle.stream());
  if (stream == nullptr) stream = 0;

  for (int candidate : candidates) {
    Config config = base_config;
    apply_candidate(config, candidate);
    op.set_config(config);

    // 先做少量预热，降低首次 launch 或 lazy init 对统计结果的扰动
    for (int i = 0; i < 2; ++i) {
      op(handle, args...);
    }

    std::vector<float> samples;
    samples.reserve(5);
    bool failed = false;

    for (int i = 0; i < 5; ++i) {
      cudaEvent_t start{};
      cudaEvent_t stop{};
      if (cudaEventCreate(&start) != cudaSuccess ||
          cudaEventCreate(&stop) != cudaSuccess) {
        failed = true;
        if (start != nullptr) cudaEventDestroy(start);
        if (stop != nullptr) cudaEventDestroy(stop);
        break;
      }
      if (cudaEventRecord(start, stream) != cudaSuccess) {
        failed = true;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        break;
      }

      op(handle, args...);

      if (cudaEventRecord(stop, stream) != cudaSuccess ||
          cudaEventSynchronize(stop) != cudaSuccess) {
        failed = true;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        break;
      }

      float elapsed_ms = 0.0f;
      if (cudaEventElapsedTime(&elapsed_ms, start, stop) != cudaSuccess) {
        failed = true;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        break;
      }

      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      samples.push_back(elapsed_ms);
    }

    if (failed || samples.empty()) continue;
    std::sort(samples.begin(), samples.end());
    measurements.emplace_back(candidate, samples[samples.size() / 2]);
  }

  if (measurements.empty()) return std::nullopt;
  std::sort(measurements.begin(), measurements.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
  return measurements.front().first;
}
#endif

}  // namespace infini::ops::detail

template <>
struct std::hash<infini::ops::detail::CacheKey> {
  std::size_t operator()(const infini::ops::detail::CacheKey& key) const {
    return key.hash;
  }
};

template <>
struct std::equal_to<infini::ops::detail::CacheKey> {
  bool operator()(const infini::ops::detail::CacheKey& a,
                  const infini::ops::detail::CacheKey& b) const {
    if (a.scalar_hash != b.scalar_hash) return false;
    if (a.tensors.size() != b.tensors.size()) return false;
    std::equal_to<infini::ops::Tensor> eq;
    for (std::size_t i = 0; i < a.tensors.size(); ++i) {
      if (!eq(a.tensors[i], b.tensors[i])) return false;
    }
    return true;
  }
};

namespace infini::ops {

template <typename Key, Device::Type kDev>
struct ActiveImplementationsImpl {
  using type = List<0>;
};

template <typename Key, Device::Type kDev>
using ActiveImplementations =
    typename ActiveImplementationsImpl<Key, kDev>::type;

class OperatorBase {
 public:
  virtual ~OperatorBase() = default;

  virtual std::size_t workspace_size_in_bytes() const { return 0; }

  void set_handle(const Handle& handle) { handle_ = handle; }

  void set_config(const Config& config) { config_ = config; }

  void set_stream(void* stream) { stream_ = stream; }

  void set_workspace(void* workspace) { workspace_ = workspace; }

  void set_workspace_size_in_bytes(std::size_t workspace_size_in_bytes) {
    workspace_size_in_bytes_ = workspace_size_in_bytes;
  }

 protected:
  Handle handle_;

  Config config_;

  void* stream_{nullptr};

  void* workspace_{nullptr};

  std::size_t workspace_size_in_bytes_{0};
};

template <typename Key, Device::Type device_type = Device::Type::kCount,
          std::size_t implementation_index = 0>
class Operator : public OperatorBase {
 public:
  template <typename... Args>
  static auto make(const Config& config, const Tensor tensor, Args&&... args) {
    std::unique_ptr<Operator> op_ptr;

    DispatchFunc<ActiveDevices<Key>>(
        tensor.device().type(),
        [&](auto device_tag) {
          constexpr Device::Type kDev = decltype(device_tag)::value;
          detail::DispatchImplementation(
              config.implementation_index(),
              [&](auto implementation_tag) {
                constexpr std::size_t kImplementationIndex =
                    decltype(implementation_tag)::value;
                if constexpr (std::is_constructible_v<
                                  Operator<Key, kDev, kImplementationIndex>,
                                  const Tensor&, Args...>) {
                  op_ptr = std::make_unique<
                      Operator<Key, kDev, kImplementationIndex>>(
                      tensor, std::forward<Args>(args)...);
                } else {
                  assert(false &&
                         "operator is not implemented for this device and "
                         "implementation index");
                }
              },
              "Operator::make(implementation_index)",
              ActiveImplementations<Key, kDev>{});
        },
        "Operator::make");

    op_ptr->set_config(config);

    return op_ptr;
  }

  template <typename... Args>
  static auto make(const Tensor tensor, Args&&... args) {
    return make({}, tensor, std::forward<Args>(args)...);
  }

  template <typename... Args>
  static auto call(const Handle& handle, const Config& config, Args&&... args) {
    static std::mutex cache_mu;
    static std::unordered_map<detail::CacheKey, std::unique_ptr<Operator>>
        cache;

    auto key = detail::CacheKey::Build(config.implementation_index(), args...);
    Operator* op_ptr{nullptr};

    {
      // 缓存容器可能被多线程并发访问，查找/插入阶段必须加锁保证安全
      std::lock_guard<std::mutex> guard(cache_mu);
      auto it{cache.find(key)};
      if (it == cache.end()) {
        it = cache.emplace(std::move(key), make(config, args...)).first;
      }
      op_ptr = it->second.get();
    }

    Config effective_config = config;
    std::string op_name = detail::OperatorNameFromTypeid(typeid(Key));
    Device::Type call_device_type = Device::Type::kCount;
    detail::ExtractFirstTensorDeviceType(&call_device_type, args...);

    auto autotune_key = detail::CacheKey::Build(op_name, call_device_type,
                                                config.implementation_index(),
                                                args...);
    autotune::Query query;
    query.key_hash = autotune_key.hash;
    query.op_name = op_name;
    query.device_type = call_device_type;
    query.implementation_index = config.implementation_index();

    const auto mode = autotune::ResolveMode(config);
    const auto cache_path = autotune::ResolveCachePath(config);
    if (!cache_path.empty()) autotune::EnsureCacheLoaded(cache_path);

    if (mode == autotune::Mode::kWarmupRecord) {
      autotune::RecordWarmupKey(query);
    }

    using AutoTunePolicy = autotune::Policy<Key>;
    if constexpr (AutoTunePolicy::kEnabled) {
      const bool support_device = AutoTunePolicy::SupportsDevice(call_device_type);
      if (support_device &&
          (mode == autotune::Mode::kTuneAndUse ||
           mode == autotune::Mode::kUseCacheOnly)) {
        query.param_name = AutoTunePolicy::ParamName();

        if (const auto cached = autotune::GetTunedInt(query); cached.has_value()) {
          if (AutoTunePolicy::ValidateIntCandidate(*cached)) {
            AutoTunePolicy::ApplyInt(effective_config, *cached);
          }
        } else if (mode == autotune::Mode::kTuneAndUse) {
#ifdef WITH_NVIDIA
          if (call_device_type == Device::Type::kNvidia) {
            const auto candidates = AutoTunePolicy::IntCandidates(effective_config);
            auto tuned = detail::TuneNvidiaIntCandidates(
                handle, *op_ptr, effective_config, candidates,
                [](Config& cfg, int candidate) {
                  AutoTunePolicy::ApplyInt(cfg, candidate);
                },
                args...);
            if (tuned.has_value() && AutoTunePolicy::ValidateIntCandidate(*tuned)) {
              autotune::SetTunedInt(query, *tuned);
              AutoTunePolicy::ApplyInt(effective_config, *tuned);
            }
          }
#endif
        }
      }
    }

    op_ptr->set_config(effective_config);
    return (*op_ptr)(handle, args...);
  }

  template <typename... Args>
  static auto call(const Tensor tensor, Args&&... args) {
    return call({}, {}, tensor, std::forward<Args>(args)...);
  }

  static std::vector<std::size_t> active_implementation_indices(
      Device::Type dev_type) {
    std::vector<std::size_t> result;
    DispatchFunc<ActiveDevices<Key>>(
        dev_type,
        [&](auto device_tag) {
          constexpr Device::Type kDev = decltype(device_tag)::value;
          result = detail::ListToVector(ActiveImplementations<Key, kDev>{});
        },
        "Operator::active_implementation_indices");
    return result;
  }

  template <typename... Args>
  auto operator()(const Handle& handle, Args&&... args) {
    set_handle(handle);
    set_stream(handle.stream());
    set_workspace(handle.workspace());
    set_workspace_size_in_bytes(handle.workspace_size_in_bytes());

    return operator()(std::forward<Args>(args)...);
  }

  template <typename... Args>
  auto operator()(Args&&... args) const {
    return (*static_cast<const Key*>(this))(std::forward<Args>(args)...);
  }

 protected:
  static constexpr Device::Type device_type_{device_type};

  static constexpr std::size_t implementation_index_{implementation_index};
};

}  // namespace infini::ops

#endif
