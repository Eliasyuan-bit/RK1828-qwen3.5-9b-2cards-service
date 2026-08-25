#include "output.h"

#include <cstdio>
#include <mutex>

#include "nlohmann/json.hpp"

namespace {
std::mutex g_stdout_mutex;
void write_line(const nlohmann::json& value)
{
  std::lock_guard<std::mutex> lock(g_stdout_mutex);
  std::printf("%s\n", value.dump().c_str());
  std::fflush(stdout);
}
}  // namespace

void write_ready_event(const std::string& model, size_t stage_count)
{
  write_line({{"ready", true}, {"model", model}, {"stage_count", stage_count}});
}
void write_delta_event(const std::string& id, const std::string& text)
{
  write_line({{"id", id}, {"event", "delta"}, {"text", text}});
}
void write_success_event(const std::string& id, const std::string& text, const GenerationMetrics& m)
{
  write_line({{"id", id}, {"ok", true}, {"text", text},
              {"metrics", {{"input_tokens", m.input_tokens}, {"output_tokens", m.output_tokens},
                           {"total_tokens", m.input_tokens + m.output_tokens}, {"ttft_ms", m.ttft_ms},
                           {"decode_ms", m.decode_ms}, {"decode_tps", m.decode_tps}}}});
}
void write_error_event(const std::string& id, const std::string& error)
{
  write_line({{"id", id}, {"ok", false}, {"error", error}});
}
