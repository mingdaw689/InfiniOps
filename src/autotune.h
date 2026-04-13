#ifndef INFINI_OPS_AUTOTUNE_H_
#define INFINI_OPS_AUTOTUNE_H_

#include <cstddef>
#include <optional>
#include <string>

#include "config.h"
#include "device.h"

namespace infini::ops::autotune {

enum class Mode {
  kDisabled = 0,
  kWarmupRecord = 1,
  kTuneAndUse = 2,
  kUseCacheOnly = 3,
};

struct Query {
  std::size_t key_hash{0};

  std::string op_name;

  Device::Type device_type{Device::Type::kCount};

  std::size_t implementation_index{0};

  std::string param_name;
};

void EnableWarmupRecord(const std::string& path);

void EnableTuneAndUse(const std::string& path);

void EnableUseCacheOnly(const std::string& path);

void Disable();

void Flush();

Mode ResolveMode(const Config& config);

std::string ResolveCachePath(const Config& config);

void EnsureCacheLoaded(const std::string& path);

void RecordWarmupKey(const Query& query);

std::optional<int> GetTunedInt(const Query& query);

void SetTunedInt(const Query& query, int value);

}  // namespace infini::ops::autotune

#endif
