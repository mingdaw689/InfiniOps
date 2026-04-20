#ifndef INFINI_OPS_CONFIG_H_
#define INFINI_OPS_CONFIG_H_

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace infini::ops {

class Config {
 public:
  std::size_t implementation_index() const { return implementation_index_; }

  void set_implementation_index(std::size_t implementation_index) {
    implementation_index_ = implementation_index;
  }

  bool autotune_enabled() const { return autotune_enabled_; }

  void set_autotune_enabled(bool autotune_enabled) {
    autotune_enabled_ = autotune_enabled;
  }

  bool autotune_record_only() const { return autotune_record_only_; }

  void set_autotune_record_only(bool autotune_record_only) {
    autotune_record_only_ = autotune_record_only;
  }

  const std::string& autotune_cache_path() const { return autotune_cache_path_; }

  void set_autotune_cache_path(std::string autotune_cache_path) {
    autotune_cache_path_ = std::move(autotune_cache_path);
  }

  // 按名称查询整数调优参数；未设置时返回空
  std::optional<int> autotune_int_param(const std::string& name) const {
    auto it = autotune_int_params_.find(name);
    if (it == autotune_int_params_.end()) return std::nullopt;
    return it->second;
  }

  // 按名称写入整数调优参数，供各算子策略在运行时应用
  void set_autotune_int_param(std::string name, int value) {
    autotune_int_params_[std::move(name)] = value;
  }

  // 当调用方希望回退默认行为时，可显式清理指定参数
  void erase_autotune_int_param(const std::string& name) {
    autotune_int_params_.erase(name);
  }

 private:
  std::size_t implementation_index_{0};

  // 默认关闭自动调优，避免影响历史行为，只有显式开启时才介入
  bool autotune_enabled_{false};

  // record_only=true 时只记录 key，不执行测速与参数选择
  bool autotune_record_only_{false};

  // 为空时表示由全局自动调优状态决定路径；非空时优先使用该路径
  std::string autotune_cache_path_;

  // 统一存放“参数名 -> 整数值”的调优结果，不再为每个算子新增专有字段
  std::unordered_map<std::string, int> autotune_int_params_;
};

}  // namespace infini::ops

#endif
