#include "request.h"

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

ChatRequest parse_chat_request(const std::string& line, int default_max_new_tokens)
{
  const nlohmann::json request = nlohmann::json::parse(line);
  ChatRequest parsed;
  parsed.id = request.value("id", "");
  parsed.max_new_tokens = request.value("max_new_tokens", default_max_new_tokens);
  parsed.enable_thinking = request.value("enable_thinking", false);
  if (request.contains("messages")) {
    parsed.prompt = build_qwen35_chat_prompt(request.at("messages"), parsed.enable_thinking);
  } else if (request.contains("prompt") && request.at("prompt").is_string()) {
    parsed.prompt = request.at("prompt").get<std::string>();
  } else {
    throw std::runtime_error("messages is required");
  }
  if (parsed.prompt.empty() || parsed.max_new_tokens <= 0) {
    throw std::runtime_error("prompt and positive max_new_tokens are required");
  }
  return parsed;
}
