#ifndef INFINI_OPS_CONFIG_H_
#define INFINI_OPS_CONFIG_H_

#include <cstddef>
#include <string>
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

  int cuda_block_size_override() const { return cuda_block_size_override_; }

  void set_cuda_block_size_override(int cuda_block_size_override) {
    cuda_block_size_override_ = cuda_block_size_override;
  }

 private:
  std::size_t implementation_index_{0};

  // 默认关闭自动调优，避免影响历史行为，只有显式开启时才介入。
  bool autotune_enabled_{false};

  // record_only=true 时只记录 key，不执行测速与参数选择。
  bool autotune_record_only_{false};

  // 为空时表示由全局自动调优状态决定路径；非空时优先使用该路径。
  std::string autotune_cache_path_;

  // 0 表示不覆盖，沿用算子原有 block size 选择逻辑。
  int cuda_block_size_override_{0};
};

}  // namespace infini::ops

#endif
