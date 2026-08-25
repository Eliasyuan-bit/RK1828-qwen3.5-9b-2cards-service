#pragma once

#include <string>

#include "nlohmann/json.hpp"

std::string build_qwen35_chat_prompt(const nlohmann::json& messages, bool enable_thinking);
