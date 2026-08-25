#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct GenerationMetrics {
  uint64_t input_tokens = 0;
  uint64_t output_tokens = 0;
  float ttft_ms = 0.0f;
  float decode_ms = 0.0f;
  float decode_tps = 0.0f;
};

void write_ready_event(const std::string& model, size_t stage_count);
void write_delta_event(const std::string& id, const std::string& text);
void write_success_event(const std::string& id, const std::string& text,
                         const GenerationMetrics& metrics);
void write_error_event(const std::string& id, const std::string& error);
