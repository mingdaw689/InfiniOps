#include "autotune.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "hash.h"

namespace infini::ops::autotune {
namespace {

struct QueryKey {
  std::size_t key_hash{0};

  std::string op_name;

  Device::Type device_type{Device::Type::kCount};

  std::size_t implementation_index{0};

  std::string param_name;

  bool operator==(const QueryKey& other) const {
    return key_hash == other.key_hash && op_name == other.op_name &&
           device_type == other.device_type &&
           implementation_index == other.implementation_index &&
           param_name == other.param_name;
  }
};

struct QueryKeyHash {
  std::size_t operator()(const QueryKey& key) const {
    std::size_t seed{0};
    HashCombine(seed, key.key_hash);
    HashCombine(seed, key.op_name);
    HashCombine(seed, key.device_type);
    HashCombine(seed, key.implementation_index);
    HashCombine(seed, key.param_name);
    return seed;
  }
};

QueryKey ToKey(const Query& query) {
  QueryKey key;
  key.key_hash = query.key_hash;
  key.op_name = query.op_name;
  key.device_type = query.device_type;
  key.implementation_index = query.implementation_index;
  key.param_name = query.param_name;
  return key;
}

class State {
 public:
  // 全局自动调优状态由单例维护，便于在任意算子调用路径共享缓存
  static State& Instance() {
    static State state;
    return state;
  }

  void Enable(Mode mode, std::string path) {
    std::lock_guard<std::mutex> guard(mu_);
    mode_ = mode;
    if (!path.empty()) cache_path_ = std::move(path);
    EnsureLoadedLocked(cache_path_);
  }

  void Disable() {
    std::lock_guard<std::mutex> guard(mu_);
    mode_ = Mode::kDisabled;
  }

  Mode mode() const {
    std::lock_guard<std::mutex> guard(mu_);
    return mode_;
  }

  std::string cache_path() const {
    std::lock_guard<std::mutex> guard(mu_);
    return cache_path_;
  }

  void EnsureLoaded(const std::string& path) {
    std::lock_guard<std::mutex> guard(mu_);
    if (!path.empty()) cache_path_ = path;
    EnsureLoadedLocked(cache_path_);
  }

  void RecordWarmupKey(const Query& query) {
    std::lock_guard<std::mutex> guard(mu_);
    recorded_keys_.insert(ToKey(query));
  }

  std::optional<int> GetTunedInt(const Query& query) const {
    std::lock_guard<std::mutex> guard(mu_);
    auto it = tuned_int_values_.find(ToKey(query));
    if (it == tuned_int_values_.end()) return std::nullopt;
    return it->second;
  }

  void SetTunedInt(const Query& query, int value) {
    std::lock_guard<std::mutex> guard(mu_);
    tuned_int_values_[ToKey(query)] = value;
    dirty_.store(true, std::memory_order_relaxed);
  }

  void Flush() {
    std::lock_guard<std::mutex> guard(mu_);
    FlushLocked();
  }

 private:
  State() { std::atexit(&State::FlushAtExit); }

  static void FlushAtExit() { Instance().Flush(); }

  void EnsureLoadedLocked(const std::string& path) {
    if (path.empty()) return;
    if (loaded_path_ == path) return;
    LoadLocked(path);
  }

  void LoadLocked(const std::string& path) {
    tuned_int_values_.clear();
    loaded_path_.clear();

    std::ifstream in(path);
    if (!in.good()) {
      loaded_path_ = path;
      return;
    }

    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      auto parsed = ParseLine(line);
      if (!parsed.has_value()) continue;
      tuned_int_values_[parsed->first] = parsed->second;
    }

    loaded_path_ = path;
  }

  void FlushLocked() {
    // 只有在有更新且路径有效时才缓存，避免无意义 IO
    if (!dirty_.load(std::memory_order_relaxed)) return;
    if (cache_path_.empty()) return;

    namespace fs = std::filesystem;
    std::error_code ec;
    auto parent = fs::path(cache_path_).parent_path();
    if (!parent.empty()) fs::create_directories(parent, ec);

    const std::string tmp_path = cache_path_ + ".tmp";
    std::ofstream out(tmp_path, std::ios::trunc);
    if (!out.good()) return;

    std::vector<std::pair<QueryKey, int>> rows;
    rows.reserve(tuned_int_values_.size());
    for (const auto& [k, v] : tuned_int_values_) {
      rows.emplace_back(k, v);
    }
    std::sort(rows.begin(), rows.end(),
              [](const auto& a, const auto& b) { return a.first.key_hash < b.first.key_hash; });

    for (const auto& [k, v] : rows) {
      out << k.key_hash << "|" << k.op_name << "|"
          << Device::StringFromType(k.device_type) << "|"
          << k.implementation_index << "|" << k.param_name << "=" << v << "\n";
    }
    out.close();
    if (!out.good()) return;

    // 同目录 rename 在大多数文件系统是原子替换，避免写入中断导致半文件
    fs::rename(tmp_path, cache_path_, ec);
    if (ec) return;
    dirty_.store(false, std::memory_order_relaxed);
  }

  static std::optional<std::pair<QueryKey, int>> ParseLine(const std::string& line) {
    std::vector<std::string> parts;
    std::string part;
    std::stringstream ss(line);
    while (std::getline(ss, part, '|')) {
      parts.push_back(part);
    }
    if (parts.size() != 5) return std::nullopt;

    try {
      QueryKey key{
      };
      key.key_hash = static_cast<std::size_t>(std::stoull(parts[0]));
      key.op_name = parts[1];
      key.device_type = Device::TypeFromString(parts[2]);
      key.implementation_index =
          static_cast<std::size_t>(std::stoull(parts[3]));

      const auto eq_pos = parts[4].find('=');
      if (eq_pos == std::string::npos || eq_pos == 0 ||
          eq_pos == parts[4].size() - 1) {
        return std::nullopt;
      }
      key.param_name = parts[4].substr(0, eq_pos);
      int value = std::stoi(parts[4].substr(eq_pos + 1));
      return std::make_pair(std::move(key), value);
    } catch (...) {
      // 缓存文件允许历史脏数据，解析失败时跳过该行即可，不阻断主流程
      return std::nullopt;
    }
  }

  mutable std::mutex mu_;

  Mode mode_{Mode::kDisabled};

  std::string cache_path_;

  std::string loaded_path_;

  std::atomic<bool> dirty_{false};

  std::unordered_set<QueryKey, QueryKeyHash> recorded_keys_;

  std::unordered_map<QueryKey, int, QueryKeyHash> tuned_int_values_;
};

}  // namespace

void EnableWarmupRecord(const std::string& path) {
  State::Instance().Enable(Mode::kWarmupRecord, path);
}

void EnableTuneAndUse(const std::string& path) {
  State::Instance().Enable(Mode::kTuneAndUse, path);
}

void EnableUseCacheOnly(const std::string& path) {
  State::Instance().Enable(Mode::kUseCacheOnly, path);
}

void Disable() { State::Instance().Disable(); }

void Flush() { State::Instance().Flush(); }

Mode ResolveMode(const Config& config) {
  // 显式 API 的全局模式优先，便于统一控制全进程行为
  const auto global_mode = State::Instance().mode();
  if (global_mode != Mode::kDisabled) return global_mode;

  // 当全局模式未开启时，允许按 Config 做每次调用级别的细粒度控制
  if (!config.autotune_enabled()) return Mode::kDisabled;
  if (config.autotune_record_only()) return Mode::kWarmupRecord;
  return Mode::kTuneAndUse;
}

std::string ResolveCachePath(const Config& config) {
  if (!config.autotune_cache_path().empty()) return config.autotune_cache_path();
  return State::Instance().cache_path();
}

void EnsureCacheLoaded(const std::string& path) {
  State::Instance().EnsureLoaded(path);
}

void RecordWarmupKey(const Query& query) { State::Instance().RecordWarmupKey(query); }

std::optional<int> GetTunedInt(const Query& query) {
  return State::Instance().GetTunedInt(query);
}

void SetTunedInt(const Query& query, int value) {
  State::Instance().SetTunedInt(query, value);
}

}  // namespace infini::ops::autotune
