#include "service_config.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>

#include "nlohmann/json.hpp"

bool load_service_config(const char* path, ServiceConfig* config)
{
  if (!path || !config) return false;
  try {
    std::ifstream stream(path);
    if (!stream) { printf("cannot open config: %s\n", path); return false; }
    nlohmann::json value;
    stream >> value;
    config->stage0_model = value.value("stage0_model", "");
    config->stage0_weight = value.value("stage0_weight", "");
    config->tokenizer = value.value("tokenizer", "");
    config->embedding = value.value("embedding", "");
    config->context_length = value.value("context_length", 0);
    config->stage_count = value.value("stage_count", 0U);
    config->bucket_size = value.value("bucket_size", 0ULL);
    const std::string core_mask = value.value("core_mask", "");
    char* end = nullptr;
    const unsigned long parsed_mask = strtoul(core_mask.c_str(), &end, 0);
    if (core_mask.empty() || !end || *end != '\0') { printf("invalid core_mask in config: %s\n", core_mask.c_str()); return false; }
    config->core_mask = static_cast<uint32_t>(parsed_mask);
    if (value.contains("device_ids")) {
      if (!value.at("device_ids").is_array()) { printf("device_ids must be an array\n"); return false; }
      for (const auto& id : value.at("device_ids")) {
        if (!id.is_string()) { printf("each device_id must be a string\n"); return false; }
        config->device_ids.push_back(id.get<std::string>());
      }
    }
    if (config->stage0_model.empty() || config->stage0_weight.empty() || config->tokenizer.empty() || config->embedding.empty() || config->context_length <= 0 || config->stage_count == 0 || config->bucket_size == 0 || config->device_ids.size() != config->stage_count) {
      printf("config requires model paths, positive context_length/stage_count/bucket_size, and one device_id per stage\n");
      return false;
    }
    return true;
  } catch (const std::exception& error) {
    printf("invalid config %s: %s\n", path, error.what());
    return false;
  }
}
