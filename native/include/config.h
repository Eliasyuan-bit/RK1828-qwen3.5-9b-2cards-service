#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ServiceConfig {
  std::string stage0_model;
  std::string stage0_weight;
  std::string tokenizer;
  std::string embedding;
  int32_t context_length = 0;
  uint32_t core_mask = 0;
  size_t stage_count = 0;
  uint64_t bucket_size = 0;
  std::vector<std::string> device_ids;
};

bool load_service_config(const char* path, ServiceConfig* config);
