#ifndef INFINI_OPS_AUTOTUNE_POLICY_H_
#define INFINI_OPS_AUTOTUNE_POLICY_H_

#include <string>
#include <vector>

#include "config.h"
#include "device.h"

namespace infini::ops {
class Add;
}

namespace infini::ops::autotune {

template <typename Op>
struct Policy {
  static constexpr bool kEnabled = false;

  static bool SupportsDevice(Device::Type) { return false; }

  static std::string ParamName() { return ""; }

  static std::vector<int> IntCandidates(const Config&) { return {}; }

  static bool ValidateIntCandidate(int) { return false; }

  static void ApplyInt(Config&, int) {}
};

template <>
struct Policy<Add> {

  // true代表开启该算子的自动调优
  static constexpr bool kEnabled = true;

  // 声明该算子在哪些设备上调优（当前只在NVIDIA）
  static bool SupportsDevice(Device::Type device_type) {
    return device_type == Device::Type::kNvidia;
  }

  // 要调优的参数
  static std::string ParamName() { return "cuda_block_size"; }

  // 候选参数值集合
  static std::vector<int> IntCandidates(const Config&) {
    return {128, 256, 512, 1024, 2048};
  }

  // 校验某个值是否是合法候选
  static bool ValidateIntCandidate(int value) {
    return value == 128 || value == 256 || value == 512 || value == 1024 ||
           value == 2048;
  }

  // 把选中的参数真正写到运行配置里
  static void ApplyInt(Config& config, int value) {
    config.set_cuda_block_size_override(value);
  }
};

}  // namespace infini::ops::autotune

#endif
