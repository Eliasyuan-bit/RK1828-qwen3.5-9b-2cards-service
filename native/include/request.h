#pragma once

#include <string>

#include "nlohmann/json.hpp"

struct ChatRequest {
  std::string id;
  std::string prompt;
  int max_new_tokens = 0;
  bool enable_thinking = false;
};

std::string build_qwen35_chat_prompt(const nlohmann::json& messages, bool enable_thinking);
ChatRequest parse_chat_request(const std::string& line, int default_max_new_tokens);
