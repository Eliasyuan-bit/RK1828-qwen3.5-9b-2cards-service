#include "request_processor.h"

#include <stdexcept>

std::string build_qwen35_chat_prompt(const nlohmann::json& messages, bool enable_thinking)
{
  if (!messages.is_array() || messages.empty()) throw std::runtime_error("messages must be a non-empty array");
  std::string prompt;
  for (const auto& message : messages) {
    const std::string role = message.value("role", "");
    if (role != "system" && role != "user" && role != "assistant") throw std::runtime_error("message role must be system, user, or assistant");
    if (!message.contains("content") || !message.at("content").is_string()) throw std::runtime_error("message content must be a string");
    prompt += "<|im_start|>" + role + "\n" + message.at("content").get<std::string>() + "<|im_end|>\n";
  }
  prompt += "<|im_start|>assistant\n";
  prompt += enable_thinking ? "<think>\n\n" : "<think>\n\n</think>\n\n";
  return prompt;
}
