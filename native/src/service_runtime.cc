// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "service_runtime.h"
#include "service_config.h"
#include "request_processor.h"

#include "Tokenizer.h"
#include "float16.h"
#include "rknn3_api.h"

#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <deque>
#include <functional>
#include <iostream>
#include <inttypes.h>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "nlohmann/json.hpp"

#define LOGW(fmt, ...) printf("\033[33m" fmt "\033[0m", ##__VA_ARGS__)
#define VLOG(fmt, ...) do { if (g_verbose) printf(fmt, ##__VA_ARGS__); } while (0)

static size_t g_stage_count = 2;
static uint64_t g_bucket_size = 128;
static bool g_verbose = false;
static bool g_ignore_eos = false;
static bool g_perf_mode = false;
static bool g_daemon_mode = false;
static std::string g_generated_text;
static std::mutex g_generated_text_mutex;
// In daemon mode the callback runs during each decode step.  Keep the active
// request id separately so it can publish JSONL deltas before final metrics.
static std::string g_stream_request_id;
static std::mutex g_stream_request_id_mutex;
static std::string g_stream_utf8_pending;
static std::mutex g_stream_utf8_mutex;

static size_t valid_utf8_prefix(const std::string& value)
{
  size_t i = 0, valid = 0;
  while (i < value.size()) {
    const unsigned char lead = static_cast<unsigned char>(value[i]);
    size_t width = 0;
    if (lead < 0x80) width = 1;
    else if (lead >= 0xC2 && lead <= 0xDF) width = 2;
    else if (lead >= 0xE0 && lead <= 0xEF) width = 3;
    else if (lead >= 0xF0 && lead <= 0xF4) width = 4;
    else { ++i; valid = i; continue; }  // invalid byte: let JSON replace it.
    if (i + width > value.size()) break;
    bool continuation = true;
    for (size_t j = 1; j < width; ++j) {
      const unsigned char byte = static_cast<unsigned char>(value[i + j]);
      continuation = continuation && ((byte & 0xC0) == 0x80);
    }
    if (!continuation) { ++i; valid = i; continue; }
    i += width;
    valid = i;
  }
  return valid;
}

static void emit_stream_delta(const std::string& piece)
{
  if (!g_daemon_mode || piece.empty()) return;
  std::string request_id;
  {
    std::lock_guard<std::mutex> lock(g_stream_request_id_mutex);
    request_id = g_stream_request_id;
  }
  if (request_id.empty()) return;
  std::string complete;
  {
    std::lock_guard<std::mutex> lock(g_stream_utf8_mutex);
    g_stream_utf8_pending += piece;
    const size_t length = valid_utf8_prefix(g_stream_utf8_pending);
    complete.assign(g_stream_utf8_pending, 0, length);
    g_stream_utf8_pending.erase(0, length);
  }
  if (complete.empty()) return;
  nlohmann::json event = {{"id", request_id}, {"event", "delta"}, {"text", complete}};
  printf("%s\n", event.dump().c_str());
  fflush(stdout);
}

struct TensorBlob
{
  rknn3_tensor_attr    attr;
  std::vector<uint8_t> data;
};

struct StageBatch
{
  std::vector<TensorBlob> tensors;
  uint64_t                n_tokens = 0;
};

struct StageSlot
{
  std::mutex              mutex;
  std::condition_variable cv;
  std::deque<StageBatch>  batches;
  uint64_t                expected_tokens = 0;
  uint64_t                emitted_tokens = 0;
  uint64_t                active_input_tokens = 0;
  bool                    producer_done = false;
  bool                    failed = false;
};

struct EmbedCallbackContext;

struct PipelineState
{
  std::vector<std::unique_ptr<StageSlot>> slots;

  explicit PipelineState(size_t stage_count)
  {
    slots.reserve(stage_count);
    for (size_t i = 0; i < stage_count; ++i) {
      slots.emplace_back(new StageSlot());
    }
  }
};

struct embedding_info
{
  int      fd = -1;
  float16* embedding_data = nullptr;
  int      embedding_dim = 0;
  int      vocab_size = 0;
};

struct EmbedCallbackContext
{
  embedding_info* embed_info = nullptr;
  PipelineState*  pipeline = nullptr;
};

struct StageCallbackContext
{
  PipelineState* pipeline = nullptr;
  size_t         stage_index = 0;
  int32_t        embedding_dim = 0;
};

struct PhasePerformance
{
  uint64_t runs = 0;
  uint64_t tokens = 0;
  double   total_ms = 0.0;
};

struct StagePerformance
{
  PhasePerformance prefill;
  PhasePerformance decode;
};

struct StageInitPerformance
{
  float rknn_init_ms = 0.0f;
  float load_model_ms = 0.0f;
  float model_init_ms = 0.0f;
  float session_init_ms = 0.0f;
  float setup_ms = 0.0f;
  float total_ms = 0.0f;
};

// Per-request statistics returned by the long-running JSONL daemon.  Keep this
// separate from the human-readable benchmark output so service callers can
// record exact token counts without parsing stderr.
struct RequestMetrics
{
  uint64_t input_tokens = 0;
  uint64_t output_tokens = 0;
  float    ttft_ms = 0.0f;
  float    decode_ms = 0.0f;
  float    decode_tps = 0.0f;
};

struct StageRuntime
{
  std::string      name;
  std::string      model_path;
  std::string      weight_path;
  rknn3_context    ctx = 0;
  rknn3_session*   session = nullptr;
  int32_t          embedding_dim = 0;
  int32_t          vocab_size = 0;
  int32_t          max_ctx_len = 0;
  rknn3_tensor*    output_tensors = nullptr;
  int              n_output_tensors = 0;
  int*             ext_input_indices = nullptr;
  int              n_ext_inputs = 0;
  StageCallbackContext callback_ctx;
  StagePerformance  performance;
  StageInitPerformance init_performance;
};

struct LastStageResultState
{
  Tokenizer* tokenizer = nullptr;
  std::mutex mutex;
  bool       has_token = false;
  int32_t    next_token = -1;
};

struct rope_cache_tensor
{
  void* data = nullptr;
  int   n_dims = 0;
  int   shape[5] = {};
  int   dtype = 0;
  int   layout = 0;
};

static const char* ROPE_CACHE_NAMES[4] = {
    "rope_cos_cache_0", "rope_sin_cache_0",
    "rope_cos_cache_1", "rope_sin_cache_1"
};

struct InputCbUserdata
{
  rope_cache_tensor rope_caches[4];
  int               rope_fd = -1;
  void*             rope_mmap_base = nullptr;
  size_t            rope_mmap_size = 0;
};

static LastStageResultState g_last_stage_result;

static size_t get_dtype_elem_size(int dtype)
{
  switch (dtype) {
  case 0:  return 4;   /* FLOAT32   */
  case 1:  return 2;   /* FLOAT16   */
  case 2:  return 1;   /* INT8      */
  case 3:  return 1;   /* UINT8     */
  case 4:  return 2;   /* INT16     */
  case 5:  return 2;   /* UINT16    */
  case 6:  return 4;   /* INT32     */
  case 7:  return 4;   /* UINT32    */
  case 8:  return 8;   /* INT64     */
  case 9:  return 8;   /* UINT64    */
  case 10: return 1;   /* BOOL      */
  case 11: return 1;   /* INT4      */
  case 12: return 1;   /* FLOAT8E4M3FN */
  case 13: return 2;   /* BFLOAT16  */
  case 14: return 1;   /* FLOAT8E8M0   */
  case 15: return 1;   /* FLOAT4E2M1   */
  default: return 1;
  }
}

static float elapsed_us(const timeval& start, const timeval& end)
{
  return (end.tv_sec - start.tv_sec) * 1e6f + (end.tv_usec - start.tv_usec);
}

static void print_performance_statistics(uint64_t prefill_tokens, float prefill_ms,
                uint64_t decode_tokens, float decode_ms)
{
  float prefill_s = prefill_ms / 1e3f;
  float prefill_tpt = prefill_tokens == 0 ? 0.0f : prefill_ms / (float)prefill_tokens;
  float prefill_tps = prefill_tokens == 0 ? 0.0f : (float)prefill_tokens / prefill_s;

  float decode_s = decode_ms / 1e3f;
  float decode_tpt = decode_tokens == 0 ? 0.0f : decode_ms / (float)decode_tokens;
  float decode_tps = decode_tokens == 0 ? 0.0f : (float)decode_tokens / decode_s;

  printf("\n\nPerformance Statistics: ");
  printf("\n-----------------------------------------------------------------------------------------\n");
  printf(" %-10s | %-16s | %-8s | %-20s | %-20s \n",
    "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second");
  printf("-----------------------------------------------------------------------------------------\n");
  printf(" %-10s | %-16.2f | %-8llu | %-20.2f | %-20.2f \n",
    "Prefill", prefill_ms, (unsigned long long)prefill_tokens, prefill_tpt, prefill_tps);
  printf(" %-10s | %-16.2f | %-8llu | %-20.2f | %-20.2f \n",
    "Decode", decode_ms, (unsigned long long)decode_tokens, decode_tpt, decode_tps);
  printf("-----------------------------------------------------------------------------------------\n");
}

static void record_stage_performance(StageRuntime& stage, bool is_prefill,
                                     uint64_t tokens, float elapsed_ms)
{
  PhasePerformance& performance = is_prefill ? stage.performance.prefill : stage.performance.decode;
  performance.runs += 1;
  performance.tokens += tokens;
  performance.total_ms += elapsed_ms;
}

static void print_per_stage_performance_statistics(const std::vector<StageRuntime>& stages)
{
  printf("\nPer-Stage Performance Statistics: \n");
  printf("----------------------------------------------------------------------------------------------------------------------\n");
  printf(" %-10s | %-10s | %-8s | %-16s | %-8s | %-20s | %-20s \n",
         "Stage", "Phase", "Runs", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second");
  printf("----------------------------------------------------------------------------------------------------------------------\n");

  for (const auto& stage : stages) {
    const PhasePerformance* phases[2] = {&stage.performance.prefill, &stage.performance.decode};
    const char* names[2] = {"Prefill", "Decode"};
    for (int i = 0; i < 2; ++i) {
      const PhasePerformance& phase = *phases[i];
      const double time_per_token = phase.tokens == 0 ? 0.0 : phase.total_ms / phase.tokens;
      const double tokens_per_second = phase.total_ms == 0.0 ? 0.0 : phase.tokens * 1000.0 / phase.total_ms;
      printf(" %-10s | %-10s | %-8llu | %-16.2f | %-8llu | %-20.2f | %-20.2f \n",
             stage.name.c_str(), names[i],
             (unsigned long long)phase.runs, phase.total_ms,
             (unsigned long long)phase.tokens, time_per_token, tokens_per_second);
    }
  }
  printf("----------------------------------------------------------------------------------------------------------------------\n");
}

static void print_model_init_statistics(const std::vector<StageRuntime>& stages, float wall_time_ms)
{
  printf("\nModel Initialization Statistics:\n");
  printf("----------------------------------------------------------------------------------------------------------------------\n");
  printf(" %-10s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s\n",
         "Stage", "rknn_init", "load_model", "model_init", "session_init", "setup", "Total (ms)");
  printf("----------------------------------------------------------------------------------------------------------------------\n");

  float total_stage_time_ms = 0.0f;
  for (const auto& stage : stages) {
    const StageInitPerformance& perf = stage.init_performance;
    total_stage_time_ms += perf.total_ms;
    printf(" %-10s | %-12.2f | %-12.2f | %-12.2f | %-12.2f | %-12.2f | %-12.2f\n",
           stage.name.c_str(), perf.rknn_init_ms, perf.load_model_ms, perf.model_init_ms,
           perf.session_init_ms, perf.setup_ms, perf.total_ms);
  }
  printf("----------------------------------------------------------------------------------------------------------------------\n");
  printf(" Multi-card initialization wall time: %.2f ms (sum of sequential stage times: %.2f ms)\n",
         wall_time_ms, total_stage_time_ms);
}

static void release_safetensors(InputCbUserdata* cb_data)
{
  if (!cb_data) return;
  if (cb_data->rope_mmap_base && cb_data->rope_mmap_base != MAP_FAILED) {
    munmap(cb_data->rope_mmap_base, cb_data->rope_mmap_size);
    cb_data->rope_mmap_base = nullptr;
  }
  if (cb_data->rope_fd >= 0) {
    close(cb_data->rope_fd);
    cb_data->rope_fd = -1;
  }
}

static void release_output_tensors(StageRuntime& stage)
{
  if (!stage.output_tensors) {
    return;
  }

  for (int i = 0; i < stage.n_output_tensors; ++i) {
    if (stage.output_tensors[i].mem) {
      rknn3_destroy_mem(stage.ctx, stage.output_tensors[i].mem);
      stage.output_tensors[i].mem = nullptr;
    }
    if (stage.output_tensors[i].attr) {
      free(stage.output_tensors[i].attr);
      stage.output_tensors[i].attr = nullptr;
    }
  }

  free(stage.output_tensors);
  stage.output_tensors = nullptr;
  stage.n_output_tensors = 0;
}

static void destroy_stage(StageRuntime& stage)
{
  if (stage.session) {
    rknn3_session_destroy(stage.session);
    stage.session = nullptr;
  }

  release_output_tensors(stage);

  if (stage.ext_input_indices) {
    free(stage.ext_input_indices);
    stage.ext_input_indices = nullptr;
  }
  stage.n_ext_inputs = 0;

  if (stage.ctx) {
    rknn3_destroy(stage.ctx);
    stage.ctx = 0;
  }
}

static void destroy_stages(std::vector<StageRuntime>& stages)
{
  for (auto& stage : stages) {
    destroy_stage(stage);
  }
}

static void release_resources(std::vector<StageRuntime>& stages, InputCbUserdata* input_cb_data,
                              embedding_info* embed_info, size_t embedding_size, Tokenizer* tokenizer)
{
  destroy_stages(stages);
  release_safetensors(input_cb_data);
  if (embed_info->embedding_data) {
    munmap(embed_info->embedding_data, embedding_size);
    embed_info->embedding_data = nullptr;
  }
  if (embed_info->fd != -1) {
    close(embed_info->fd);
    embed_info->fd = -1;
  }
  delete tokenizer;
}

static void reset_stage_slot(StageSlot& slot)
{
  std::lock_guard<std::mutex> lock(slot.mutex);
  slot.batches.clear();
  slot.expected_tokens = 0;
  slot.emitted_tokens = 0;
  slot.active_input_tokens = 0;
  slot.producer_done = false;
  slot.failed = false;
}

static void reset_pipeline(PipelineState& pipeline)
{
  for (auto& slot : pipeline.slots) {
    reset_stage_slot(*slot);
  }
}

static void close_stage_slot(StageSlot& slot)
{
  {
    std::lock_guard<std::mutex> lock(slot.mutex);
    slot.producer_done = true;
  }
  slot.cv.notify_all();
}

static void fail_pipeline(PipelineState& pipeline)
{
  for (auto& slot_ptr : pipeline.slots) {
    StageSlot& slot = *slot_ptr;
    {
      std::lock_guard<std::mutex> lock(slot.mutex);
      slot.failed = true;
    }
    slot.cv.notify_all();
  }
}

static bool pipeline_failed(PipelineState& pipeline)
{
  for (auto& slot_ptr : pipeline.slots) {
    std::lock_guard<std::mutex> lock(slot_ptr->mutex);
    if (slot_ptr->failed) {
      return true;
    }
  }
  return false;
}

static void reset_last_stage_result()
{
  std::lock_guard<std::mutex> lock(g_last_stage_result.mutex);
  g_last_stage_result.has_token = false;
  g_last_stage_result.next_token = -1;
}

static bool get_last_stage_token(int32_t* token)
{
  std::lock_guard<std::mutex> lock(g_last_stage_result.mutex);
  if (!g_last_stage_result.has_token) {
    return false;
  }
  *token = g_last_stage_result.next_token;
  return true;
}

static bool name_contains(const char* name, const char* needle)
{
  return name && needle && strstr(name, needle) != nullptr;
}

static void dump_tensor_blob(const TensorBlob& blob, size_t index, const char* prefix)
{
  VLOG("%s tensor[%zu]: name=%s, dtype=%d, n_elems=%u, aligned_size=%llu\n",
       prefix,
       index,
       blob.attr.name,
       (int)blob.attr.dtype,
       blob.attr.n_elems,
       (unsigned long long)blob.attr.aligned_size);
}

static const TensorBlob* pick_embed_tensor(const std::vector<TensorBlob>& tensors)
{
  if (tensors.empty()) {
    return nullptr;
  }

  for (const auto& tensor : tensors) {
    if (name_contains(tensor.attr.name, "hidden") || name_contains(tensor.attr.name, "last_hidden") ||
        name_contains(tensor.attr.name, "output")) {
      return &tensor;
    }
  }

  return &tensors.front();
}

static int tokenizer_callback(void* userdata, const char* text, int32_t text_len, int32_t* tokens, int32_t n_tokens_max)
{
  Tokenizer* tokenizer = (Tokenizer*)userdata;
  if (!tokenizer || !text || !tokens || n_tokens_max <= 0) {
    return -1;
  }

  int n_tokens = tokenizer->Tokenize(text, text_len, tokens, n_tokens_max);
  VLOG("[tokenizer_callback] text=%s, text_len=%d, n_tokens=%d\n", text, text_len, n_tokens);
  if (n_tokens <= 0) {
    printf("tokenizer failed for input text\n");
  }
  return n_tokens;
}

static int embed_callback(void* userdata, int32_t* tokens, uint64_t num_tokens, void* embed, uint64_t len)
{
  EmbedCallbackContext* ctx = (EmbedCallbackContext*)userdata;
  embedding_info* info = ctx ? ctx->embed_info : nullptr;
  if (!info || !tokens || !embed || info->embedding_dim <= 0 || !info->embedding_data) {
    return -1;
  }

  if (len != num_tokens * (uint64_t)info->embedding_dim * sizeof(float16)) {
    printf("invalid embed buffer size\n");
    return -1;
  }

  for (uint64_t n = 0; n < num_tokens; ++n) {
    memcpy((unsigned char*)embed + n * info->embedding_dim * sizeof(float16),
           info->embedding_data + tokens[n] * info->embedding_dim,
           info->embedding_dim * sizeof(float16));
  }

  // 统计 stage0 输入 token 数，用于 prefill 性能统计。
  if (ctx->pipeline && !ctx->pipeline->slots.empty()) {
    StageSlot& slot = *ctx->pipeline->slots.front();
    std::lock_guard<std::mutex> lock(slot.mutex);
    slot.expected_tokens += num_tokens;
    VLOG("[embed_callback] num_tokens=%llu, total=%llu, token_id=%d\n",
    (unsigned long long)num_tokens, (unsigned long long)slot.expected_tokens, tokens[num_tokens - 1]);
  }

  return 0;
}

static int result_callback(void* userdata, RKLLMResult* result, LLMCallState state)
{
  LastStageResultState* result_state = (LastStageResultState*)userdata;
  Tokenizer* tokenizer = result_state ? result_state->tokenizer : nullptr;

  if (state == RKLLM_RUN_NORMAL && result && tokenizer) {
    int32_t next_token = -1;
    if (result->num_tokens > 0) {
      next_token = result->token_ids[result->num_tokens - 1];
      std::lock_guard<std::mutex> lock(result_state->mutex);
      result_state->next_token = next_token;
      result_state->has_token = true;
    }

    if (!g_perf_mode) {
      std::string piece;
      if (result->num_tokens == 1) {
        piece = tokenizer->TokenToPiece(result->token_ids[0]);
      } else {
        piece = tokenizer->Decode(result->token_ids, result->num_tokens);
      }
      {
        std::lock_guard<std::mutex> lock(g_generated_text_mutex);
        g_generated_text += piece;
      }
      emit_stream_delta(piece);
      VLOG("[result_callback] %s, next_token=%d\n", piece.c_str(), next_token);
      if (!g_daemon_mode) {
        printf("%s", piece.c_str());
        fflush(stdout);
      }
    }
  }

  return 0;
}

static int stage_output_callback(void* userdata, rknn3_tensor* output_tensors, uint32_t n_output_tensors, LLMOutputCallbackState state)
{
  auto* cb_ctx = reinterpret_cast<StageCallbackContext*>(userdata);
  if (!cb_ctx || !cb_ctx->pipeline || cb_ctx->stage_index >= cb_ctx->pipeline->slots.size()) {
    return -1;
  }

  StageSlot& slot = *cb_ctx->pipeline->slots[cb_ctx->stage_index];

  VLOG("[Stage %zu] output_callback: state=%d, n_outputs=%u\n", cb_ctx->stage_index, state, n_output_tensors);

  StageBatch batch;
  batch.tensors.reserve(n_output_tensors);

  for (uint32_t i = 0; i < n_output_tensors; ++i) {
    if (!output_tensors[i].attr || !output_tensors[i].mem || !output_tensors[i].mem->virt_addr) {
      continue;
    }

    TensorBlob blob;
    blob.attr = *output_tensors[i].attr;
    blob.data.resize((size_t)blob.attr.aligned_size);
    memcpy(blob.data.data(), output_tensors[i].mem->virt_addr, blob.data.size());
    dump_tensor_blob(blob, i, "  [captured]");
    batch.tensors.push_back(std::move(blob));
  }

  if (!batch.tensors.empty()) {
    {
      std::lock_guard<std::mutex> lock(slot.mutex);
      uint64_t remaining_tokens = slot.expected_tokens > slot.emitted_tokens
                                      ? slot.expected_tokens - slot.emitted_tokens
                                      : 0;
      if (slot.active_input_tokens > 0) {
        batch.n_tokens = slot.active_input_tokens;
      } else if (state == RKLLM_OUTPUT_CALLBACK_PREFILL_FINISHED) {
        batch.n_tokens = remaining_tokens;
      } else {
        batch.n_tokens = remaining_tokens > g_bucket_size ? g_bucket_size : remaining_tokens;
      }

      if (batch.n_tokens == 0) {
        const TensorBlob* embed = pick_embed_tensor(batch.tensors);
        if (embed && embed->attr.n_elems > 0 && cb_ctx->embedding_dim > 0) {
          batch.n_tokens = embed->attr.n_elems / (uint64_t)cb_ctx->embedding_dim;
        }
      }
      slot.emitted_tokens += batch.n_tokens;
      slot.batches.push_back(std::move(batch));
    }
    slot.cv.notify_one();
  }

  return 0;
}

static int input_callback(void* userdata, rknn3_tensor* input_tensors, uint32_t n_input_tensors,
                          LLMInputCallbackParam param)
{
  InputCbUserdata* cb_data = (InputCbUserdata*)userdata;

  for (uint32_t i = 0; i < n_input_tensors; ++i) {
    bool handled_as_rope = false;
    for (int c = 0; c < 4; c++) {
      if (strcmp(input_tensors[i].attr->name, ROPE_CACHE_NAMES[c]) == 0) {
        const rope_cache_tensor* cache     = &cb_data->rope_caches[c];
        const size_t             elem_sz   = get_dtype_elem_size(cache->dtype);
        const int                C1        = cache->shape[1];
        const size_t             c2_bytes  = (size_t)cache->shape[4] * elem_sz;
        const size_t             src_stride = (size_t)cache->shape[3] * c2_bytes;
        const size_t             dst_stride = (size_t)input_tensors[i].attr->shape[3] * c2_bytes;
        // 取 min 防止 src 越界读取
        const size_t             copy_stride = src_stride < dst_stride ? src_stride : dst_stride;
        const uint8_t*           src = (const uint8_t*)cache->data
                                       + (size_t)param.pos * c2_bytes;
        uint8_t*                 dst = (uint8_t*)input_tensors[i].mem->virt_addr;
        for (int c1 = 0; c1 < C1; c1++, src += src_stride, dst += dst_stride) {
          memcpy(dst, src, copy_stride);
        }
        handled_as_rope = true;
        break;
      }
    }
    if (handled_as_rope) continue;
  }

  return 0;
}

static int load_safetensors(const char* path, rope_cache_tensor caches[4],
                            int* fd_out, void** mmap_base_out, size_t* mmap_size_out)
{
  int         fd          = -1;
  void*       map         = MAP_FAILED;
  uint64_t    header_size = 0;
  struct stat st;
  int         ret         = -1;

  fd = open(path, O_RDONLY);
  if (fd < 0) {
    printf("Failed to open safetensors file: %s\n", path);
    goto err;
  }
  if (fstat(fd, &st) < 0) {
    printf("Failed to stat safetensors file: %s\n", path);
    goto err;
  }

  if (read(fd, &header_size, 8) != 8) {
    printf("Failed to read safetensors header size\n");
    goto err;
  }
  if (header_size == 0 || header_size > (uint64_t)st.st_size - 8) {
    printf("Invalid safetensors header size: %" PRIu64 "\n", header_size);
    goto err;
  }

  map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) {
    printf("Failed to mmap safetensors file: %s\n", path);
    goto err;
  }

  {
    const char*    json_ptr  = (const char*)map + 8;
    const uint8_t* data_base = (const uint8_t*)map + 8 + header_size;
    try {
      nlohmann::json j = nlohmann::json::parse(json_ptr, json_ptr + header_size);

      nlohmann::json meta_index = nlohmann::json::parse(
          j.at("__metadata__").at("index").get<std::string>());

      ret = 0;
      for (int i = 0; i < 4; i++) {
        const auto& meta_t = meta_index.at(ROPE_CACHE_NAMES[i]);
        int         dtype  = meta_t.at("dtype").get<int>();
        int         layout = meta_t.at("layout").get<int>();

        const auto& t      = j.at(ROPE_CACHE_NAMES[i]);
        auto shape_v   = t.at("shape").get<std::vector<int>>();
        auto offsets_v = t.at("data_offsets").get<std::vector<int64_t>>();

        int n_dims = (int)shape_v.size();
        if (n_dims != 5 || layout != 3) {
          printf("Tensor '%s': expected 5-D NC1HWC2 (layout=%d, n_dims=%d)\n",
                 ROPE_CACHE_NAMES[i], layout, n_dims);
          ret = -1;
          break;
        }
        caches[i].data   = (void*)(data_base + offsets_v[0]);
        caches[i].n_dims = n_dims;
        caches[i].dtype  = dtype;
        caches[i].layout = layout;
        for (int d = 0; d < n_dims; d++) caches[i].shape[d] = shape_v[d];
        printf("Loaded %-24s  dtype=%-2d  shape=[%d,%d,%d,%d,%d]\n",
               ROPE_CACHE_NAMES[i], dtype,
               caches[i].shape[0], caches[i].shape[1], caches[i].shape[2],
               caches[i].shape[3], caches[i].shape[4]);
      }
    } catch (const nlohmann::json::exception& e) {
      printf("Failed to parse safetensors JSON: %s\n", e.what());
      ret = -1;
    }
  }

err:
  if (ret != 0) {
    if (map != MAP_FAILED) munmap(map, (size_t)st.st_size);
    if (fd >= 0) close(fd);
    return ret;
  }
  *fd_out        = fd;
  *mmap_base_out = map;
  *mmap_size_out = (size_t)st.st_size;
  return 0;
}

static int init_tokenizer_and_embedding(const char* tokenizer_path, const char* embedding_path, VocabInfo* vocab_info,
                                        Tokenizer** tokenizer, embedding_info* embed_info, struct stat* emb_st)
{
  *tokenizer = new Tokenizer(TOKENIZER_BACKEND_LLAMA, tokenizer_path);
  (*tokenizer)->GetVocabInfo(vocab_info);

  embed_info->fd = open(embedding_path, O_RDONLY);
  if (embed_info->fd == -1) {
    printf("Failed to open embedding file: %s\n", embedding_path);
    delete *tokenizer;
    *tokenizer = nullptr;
    return -1;
  }

  if (fstat(embed_info->fd, emb_st) == -1) {
    printf("Failed to get embedding file size\n");
    close(embed_info->fd);
    embed_info->fd = -1;
    delete *tokenizer;
    *tokenizer = nullptr;
    return -1;
  }

  embed_info->embedding_data = (float16*)mmap(NULL, emb_st->st_size, PROT_READ, MAP_PRIVATE, embed_info->fd, 0);
  if (embed_info->embedding_data == MAP_FAILED) {
    printf("Failed to mmap embedding file\n");
    embed_info->embedding_data = nullptr;
    close(embed_info->fd);
    embed_info->fd = -1;
    delete *tokenizer;
    *tokenizer = nullptr;
    return -1;
  }

  embed_info->vocab_size = vocab_info->vocab_size;
  embed_info->embedding_dim = (emb_st->st_size / vocab_info->vocab_size) / sizeof(float16);
  return 0;
}

static int init_output_tensors(StageRuntime& stage)
{
  rknn3_input_output_num io_num;
  memset(&io_num, 0, sizeof(io_num));
  int ret = rknn3_query(stage.ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN3_SUCCESS) {
    printf("[%s] query io num failed, ret=%d\n", stage.name.c_str(), ret);
    return -1;
  }

  stage.n_output_tensors = (int)io_num.n_output;
  stage.output_tensors = (rknn3_tensor*)calloc(io_num.n_output, sizeof(rknn3_tensor));
  if (!stage.output_tensors) {
    return -1;
  }

  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    stage.output_tensors[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
    if (!stage.output_tensors[i].attr) {
      // 释放前序已分配的 attr 和 output_tensors
      for (uint32_t j = 0; j < i; ++j) {
        if (stage.output_tensors[j].mem) {
          rknn3_destroy_mem(stage.ctx, stage.output_tensors[j].mem);
          stage.output_tensors[j].mem = nullptr;
        }
        free(stage.output_tensors[j].attr);
        stage.output_tensors[j].attr = nullptr;
      }
      free(stage.output_tensors);
      stage.output_tensors = nullptr;
      stage.n_output_tensors = 0;
      return -1;
    }
    memset(stage.output_tensors[i].attr, 0, sizeof(rknn3_tensor_attr));
    stage.output_tensors[i].attr->index = i;

    ret = rknn3_query(stage.ctx, RKNN3_QUERY_OUTPUT_ATTR, stage.output_tensors[i].attr, sizeof(rknn3_tensor_attr));
    if (ret != RKNN3_SUCCESS) {
      printf("[%s] query output attr[%u] failed, ret=%d\n", stage.name.c_str(), i, ret);
      // 释放前序已分配的资源
      for (uint32_t j = 0; j < i; ++j) {
        if (stage.output_tensors[j].mem) {
          rknn3_destroy_mem(stage.ctx, stage.output_tensors[j].mem);
          stage.output_tensors[j].mem = nullptr;
        }
        free(stage.output_tensors[j].attr);
        stage.output_tensors[j].attr = nullptr;
      }
      free(stage.output_tensors[i].attr);
      stage.output_tensors[i].attr = nullptr;
      free(stage.output_tensors);
      stage.output_tensors = nullptr;
      stage.n_output_tensors = 0;
      return -1;
    }

    stage.output_tensors[i].mem = rknn3_create_mem(stage.ctx,
                                                    stage.output_tensors[i].attr->aligned_size,
                                                    stage.output_tensors[i].attr->core_id,
                                                    RKNN3_FLAG_MEMORY_CACHEABLE);
    if (!stage.output_tensors[i].mem) {
      printf("[%s] create output mem[%u] failed\n", stage.name.c_str(), i);
      // 释放前序已分配的资源
      for (uint32_t j = 0; j < i; ++j) {
        if (stage.output_tensors[j].mem) {
          rknn3_destroy_mem(stage.ctx, stage.output_tensors[j].mem);
          stage.output_tensors[j].mem = nullptr;
        }
        free(stage.output_tensors[j].attr);
        stage.output_tensors[j].attr = nullptr;
      }
      free(stage.output_tensors[i].attr);
      stage.output_tensors[i].attr = nullptr;
      free(stage.output_tensors);
      stage.output_tensors = nullptr;
      stage.n_output_tensors = 0;
      return -1;
    }
  }

  return 0;
}

static int query_ext_input_indices(StageRuntime& stage)
{
  rknn3_input_output_num io_num;
  memset(&io_num, 0, sizeof(io_num));
  int ret = rknn3_query(stage.ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN3_SUCCESS) {
    printf("[%s] query io num for ext inputs failed, ret=%d\n", stage.name.c_str(), ret);
    return -1;
  }

  int n_ext = 0;
  for (uint32_t i = 0; i < io_num.n_input; ++i) {
    rknn3_tensor_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.index = i;
    ret = rknn3_query(stage.ctx, RKNN3_QUERY_INPUT_ATTR, &attr, sizeof(attr));
    if (ret != RKNN3_SUCCESS) {
      printf("[%s] query input attr[%u] failed, ret=%d\n", stage.name.c_str(), i, ret);
      return -1;
    }
    if (strcmp(attr.name, "per_layer_inputs") == 0) {
      n_ext++;
    } else if (strstr(attr.name, "rope_cos_cache") || strstr(attr.name, "rope_sin_cache")) {
      n_ext++;
    }
  }

  if (n_ext == 0) {
    return 0;
  }

  stage.ext_input_indices = (int*)malloc(n_ext * sizeof(int));
  if (!stage.ext_input_indices) {
    printf("[%s] malloc ext_input_indices failed\n", stage.name.c_str());
    return -1;
  }
  stage.n_ext_inputs = 0;

  for (uint32_t i = 0; i < io_num.n_input; ++i) {
    rknn3_tensor_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.index = i;
    ret = rknn3_query(stage.ctx, RKNN3_QUERY_INPUT_ATTR, &attr, sizeof(attr));
    if (ret != RKNN3_SUCCESS) {
      return -1;
    }
    if (strcmp(attr.name, "per_layer_inputs") == 0) {
      stage.ext_input_indices[stage.n_ext_inputs++] = (int)i;
    } else if (strstr(attr.name, "rope_cos_cache") || strstr(attr.name, "rope_sin_cache")) {
      stage.ext_input_indices[stage.n_ext_inputs++] = (int)i;
    }
  }

  printf("[%s] found %d ext input tensors (per_layer_inputs/rope_caches)\n", stage.name.c_str(), stage.n_ext_inputs);
  return 0;
}

static bool init_stage(StageRuntime& stage, PipelineState& pipeline, size_t stage_idx, const char* device_id,
                       const char* model_path, const char* weight_path, const rknn3_llm_param& session_param,
                       uint32_t run_core_mask,
                       bool is_last_stage, Tokenizer* tokenizer,
                       EmbedCallbackContext* embed_ctx,
                       InputCbUserdata* input_cb_data)
{
  timeval stage_start;
  timeval step_start;
  timeval step_end;
  gettimeofday(&stage_start, NULL);
  stage.model_path = model_path;
  stage.weight_path = weight_path;
  stage.callback_ctx.pipeline = &pipeline;
  stage.callback_ctx.stage_index = stage_idx;

  printf("[%s] init stage: model=%s, weight=%s, device_id=%s\n",
         stage.name.c_str(), model_path, weight_path, device_id);

  rknn3_init_extend ext;
  memset(&ext, 0, sizeof(ext));
  ext.device_id = const_cast<char*>(device_id);

  gettimeofday(&step_start, NULL);
  int ret = rknn3_init(&stage.ctx, &ext);
  gettimeofday(&step_end, NULL);
  stage.init_performance.rknn_init_ms = elapsed_us(step_start, step_end) / 1e3f;
  if (ret != RKNN3_SUCCESS) {
    printf("[%s] rknn3_init failed, ret=%d\n", stage.name.c_str(), ret);
    return false;
  }

  gettimeofday(&step_start, NULL);
  ret = rknn3_load_model_from_path(stage.ctx, stage.model_path.c_str(), stage.weight_path.c_str());
  gettimeofday(&step_end, NULL);
  stage.init_performance.load_model_ms = elapsed_us(step_start, step_end) / 1e3f;
  if (ret != RKNN3_SUCCESS) {
    printf("[%s] rknn3_load_model_from_path failed, ret=%d\n", stage.name.c_str(), ret);
    destroy_stage(stage);
    return false;
  }

  rknn3_config cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.run_core_mask = run_core_mask;
  gettimeofday(&step_start, NULL);
  ret = rknn3_model_init(stage.ctx, &cfg);
  gettimeofday(&step_end, NULL);
  stage.init_performance.model_init_ms = elapsed_us(step_start, step_end) / 1e3f;
  if (ret != RKNN3_SUCCESS) {
    printf("[%s] rknn3_model_init failed, ret=%d\n", stage.name.c_str(), ret);
    destroy_stage(stage);
    return false;
  }
  // rknn3_set_input_name_alias(stage.ctx, "inputs_embeds", "input_embeds");
  // rknn3_set_input_name_alias(stage.ctx, "hidden_states", "input_embeds");
  rknn3_llm_config llm_cfg;
  memset(&llm_cfg, 0, sizeof(llm_cfg));
  ret = rknn3_query(stage.ctx, RKNN3_QUERY_LLM_CONFIG, &llm_cfg, sizeof(llm_cfg));
  if (ret != RKNN3_SUCCESS) {
    printf("[%s] query llm config failed, ret=%d\n", stage.name.c_str(), ret);
    destroy_stage(stage);
    return false;
  }
  stage.embedding_dim = (int32_t)llm_cfg.embedding_dim;
  stage.vocab_size = (int32_t)llm_cfg.vocab_size;
  stage.max_ctx_len = (int32_t)llm_cfg.max_ctx_len;
  stage.callback_ctx.embedding_dim = stage.embedding_dim;

  if (init_output_tensors(stage) != 0) {
    printf("[%s] init output tensors failed\n", stage.name.c_str());
    destroy_stage(stage);
    return false;
  }

  rknn3_llm_param local_param = session_param;

  gettimeofday(&step_start, NULL);
  stage.session = rknn3_session_init(stage.ctx, &local_param, 1);
  gettimeofday(&step_end, NULL);
  stage.init_performance.session_init_ms = elapsed_us(step_start, step_end) / 1e3f;
  if (!stage.session) {
    printf("[%s] rknn3_session_init failed\n", stage.name.c_str());
    destroy_stage(stage);
    return false;
  }
  rknn3_session_set_chat_template(stage.session, "", "", "");
  RKLLMCallback callback;
  memset(&callback, 0, sizeof(callback));

  if (!is_last_stage) {
    callback.output_callback = stage_output_callback;
    callback.output_userdata = &stage.callback_ctx;
    callback.output_tensors = stage.output_tensors;
    callback.n_output_tensors = stage.n_output_tensors;
  }

  callback.tokenizer_callback = tokenizer_callback;
  callback.tokenizer_userdata = tokenizer;
  callback.embed_callback = embed_callback;
  callback.embed_userdata = embed_ctx;
  callback.result_callback = result_callback;
  callback.result_userdata = &g_last_stage_result;

  // 如果提供了 safetensors (rope caches)，则注册 input_callback 并设置 ext input indices
  if (input_cb_data && input_cb_data->rope_mmap_base) {
    if (query_ext_input_indices(stage) != 0) {
      printf("[%s] query_ext_input_indices failed\n", stage.name.c_str());
      destroy_stage(stage);
      return false;
    }
    callback.input_callback = input_callback;
    callback.input_userdata = input_cb_data;
    callback.input_tensors_index = stage.ext_input_indices;
    callback.n_input_tensors     = stage.n_ext_inputs;
    for (int i = 0; i < stage.n_ext_inputs; ++i) {
      VLOG("[%s] ext input tensor[%d] index=%d\n", stage.name.c_str(), i, stage.ext_input_indices[i]);
    }
  }

  gettimeofday(&step_start, NULL);
  ret = rknn3_session_set_callback(stage.session, &callback);
  gettimeofday(&step_end, NULL);
  stage.init_performance.setup_ms = elapsed_us(step_start, step_end) / 1e3f;
  if (ret != RKNN3_SUCCESS) {
    printf("[%s] rknn3_session_set_callback failed, ret=%d\n", stage.name.c_str(), ret);
    destroy_stage(stage);
    return false;
  }

  printf("[%s] init done: embedding_dim=%d vocab_size=%d max_ctx_len=%d outputs=%d\n",
         stage.name.c_str(), stage.embedding_dim, stage.vocab_size, stage.max_ctx_len, stage.n_output_tensors);
  timeval stage_end;
  gettimeofday(&stage_end, NULL);
  stage.init_performance.total_ms = elapsed_us(stage_start, stage_end) / 1e3f;
  return true;
}

static bool wait_stage_batch(StageSlot& slot, StageBatch* batch)
{
  std::unique_lock<std::mutex> lock(slot.mutex);
  slot.cv.wait(lock, [&slot]() {
    return !slot.batches.empty() || slot.producer_done || slot.failed;
  });

  if (slot.failed || slot.batches.empty()) {
    return false;
  }
  *batch = std::move(slot.batches.front());
  slot.batches.pop_front();
  return true;
}

static void run_stage_worker(size_t stage_idx, std::vector<StageRuntime>& stages,
                             PipelineState& pipeline, const rknn3_llm_infer_param& infer_param,
                             bool is_prefill)
{
  StageRuntime& stage = stages[stage_idx];
  StageSlot& input_slot = *pipeline.slots[stage_idx - 1];
  StageSlot* output_slot = stage_idx + 1 < stages.size() ? pipeline.slots[stage_idx].get() : nullptr;
  bool is_last_stage = (stage_idx == stages.size() - 1);
  uint64_t consumed_tokens = 0;

  while (true) {
    StageBatch batch;
    if (!wait_stage_batch(input_slot, &batch)) {
      std::lock_guard<std::mutex> lock(input_slot.mutex);
      if (!input_slot.failed && input_slot.producer_done && input_slot.batches.empty()) {
        break;
      }
      if (output_slot) {
        close_stage_slot(*output_slot);
      }
      return;
    }

    const TensorBlob* embed = pick_embed_tensor(batch.tensors);
    if (!embed || batch.n_tokens == 0 || stage.embedding_dim <= 0) {
      printf("[stage%zu] invalid pipeline batch\n", stage_idx);
      fail_pipeline(pipeline);
      return;
    }

    size_t embed_bytes = batch.n_tokens * (uint64_t)stage.embedding_dim * sizeof(float16);
    if (embed_bytes > embed->data.size()) {
      printf("[stage%zu] embed buffer too small: need=%zu, got=%zu\n",
             stage_idx, embed_bytes, embed->data.size());
      fail_pipeline(pipeline);
      return;
    }

    if (output_slot) {
      std::lock_guard<std::mutex> lock(output_slot->mutex);
      output_slot->expected_tokens += batch.n_tokens;
      output_slot->active_input_tokens = batch.n_tokens;
    }

    rknn3_llm_input embed_input;
    memset(&embed_input, 0, sizeof(embed_input));
    embed_input.input_type = RKNN3_LLM_INPUT_EMBED;
    embed_input.llm_input.embed = (float16*)embed->data.data();
    embed_input.llm_input.n_tokens = batch.n_tokens;
    rknn3_llm_infer_param local_param = infer_param;
    if (is_last_stage) {
      // 最后一段：根据 embed_callback 中统计的总 token 数判断是否最后一桶
      uint64_t total_tokens;
      {
        std::lock_guard<std::mutex> lock(pipeline.slots[0]->mutex);
        total_tokens = pipeline.slots[0]->expected_tokens;
      }
      local_param.disable_sampling = (consumed_tokens + batch.n_tokens < total_tokens);
    }
    consumed_tokens += batch.n_tokens;
    VLOG("[stage%zu] consume batch: n_tokens=%llu, bytes=%zu, disable_sampling=%d\n",
         stage_idx, (unsigned long long)batch.n_tokens, embed_bytes,
         (int)local_param.disable_sampling);
    timeval run_start;
    timeval run_end;
    gettimeofday(&run_start, NULL);
    int ret = rknn3_session_run(stage.session, &embed_input, 1, &local_param);
    gettimeofday(&run_end, NULL);

    if (output_slot) {
      std::lock_guard<std::mutex> lock(output_slot->mutex);
      output_slot->active_input_tokens = 0;
    }
    if (ret != RKNN3_SUCCESS) {
      printf("[stage%zu] run failed ret=%d\n", stage_idx, ret);
      fail_pipeline(pipeline);
      return;
    }
    record_stage_performance(stage, is_prefill, batch.n_tokens,
                             elapsed_us(run_start, run_end) / 1e3f);
  }

  if (output_slot) {
    close_stage_slot(*output_slot);
  }
}

static bool run_pipeline_once(std::vector<StageRuntime>& stages, PipelineState& pipeline,
                              const char* prompt, const std::vector<int32_t>* input_tokens,
                              uint64_t* stage0_input_tokens, bool is_prefill,
                              bool enable_thinking = false)
{
  if (stages.empty()) {
    return false;
  }

  rknn3_llm_infer_param infer_param;
  memset(&infer_param, 0, sizeof(infer_param));
  infer_param.keep_history = 1;
  infer_param.max_new_tokens = 1;
  // 使用 prefill_only 模式进行单步推理
  infer_param.prefill_only = true;
  // 只有最后一段才进行采样，且prefill时只有最后一桶结束才采样
  infer_param.disable_sampling = true;

  reset_pipeline(pipeline);
  reset_last_stage_result();

  rknn3_llm_input first_input;
  memset(&first_input, 0, sizeof(first_input));
  if (prompt) {
    first_input.input_type = RKNN3_LLM_INPUT_PROMPT;
    first_input.llm_input.prompt = prompt;
    first_input.llm_input.enable_thinking = enable_thinking;
  } else {
    if (!input_tokens || input_tokens->empty()) {
      return false;
    }
    first_input.input_type = RKNN3_LLM_INPUT_TOKEN;
    first_input.llm_input.tokens = const_cast<int32_t*>(input_tokens->data());
    first_input.llm_input.n_tokens = input_tokens->size();
  }

  std::vector<std::thread> workers;
  workers.reserve(stages.size() - 1);
  for (size_t i = 1; i < stages.size(); ++i) {
    workers.emplace_back(run_stage_worker, i, std::ref(stages), std::ref(pipeline),
                         std::cref(infer_param), is_prefill);
  }

  timeval stage0_start;
  timeval stage0_end;
  gettimeofday(&stage0_start, NULL);
  int ret = rknn3_session_run(stages[0].session, &first_input, 1, &infer_param);
  gettimeofday(&stage0_end, NULL);
  if (ret != RKNN3_SUCCESS) {
    printf("[stage0] run failed ret=%d\n", ret);
    fail_pipeline(pipeline);
  }
  close_stage_slot(*pipeline.slots[0]);

  for (auto& worker : workers) {
    worker.join();
  }

  uint64_t stage0_tokens = 0;
  {
    std::lock_guard<std::mutex> lock(pipeline.slots[0]->mutex);
    stage0_tokens = pipeline.slots[0]->expected_tokens;
  }
  if (ret == RKNN3_SUCCESS) {
    record_stage_performance(stages[0], is_prefill, stage0_tokens,
                             elapsed_us(stage0_start, stage0_end) / 1e3f);
  }
  if (stage0_input_tokens) {
    *stage0_input_tokens = stage0_tokens;
  }

  return ret == RKNN3_SUCCESS && !pipeline_failed(pipeline);
}


bool ServiceRuntime::init(int argc, char** argv)
{
  argc_ = argc;
  argv_ = argv;
  return argc_ > 0 && argv_ != nullptr;
}

void ServiceRuntime::deinit()
{
  // All RKNN, Tokenizer, mmap and KV-cache resources are released by run()
  // before it returns. This method keeps the process lifecycle explicit for
  // callers and remains the single extension point for future persistent state.
}

int ServiceRuntime::run()
{
  const int argc = argc_;
  char** argv = argv_;
  const bool config_mode = argc >= 3 && strcmp(argv[1], "--config") == 0;
  if (!config_mode && argc < 9) {
    LOGW("Usage: %s <stage0_model.rknn> <stage0_weight> <tokenizer.gguf> <embedding.bin> <max_context_len> <run_core_mask> <stage_count> <bucket_size> [prompt] [max_new_tokens] [verbose] [ignore_eos] [rope_path] [device_id_0] ... [--perf <input_tokens> <output_tokens>]\n",
         argv[0]);
    LOGW("   or: %s --config <service.json> [--daemon]\n", argv[0]);
    return -1;
  }

  ServiceConfig service_config;
  std::vector<std::string> configured_device_ids;
  const char* base_model_path = nullptr;
  const char* base_weight_path = nullptr;
  const char* tokenizer_path = nullptr;
  const char* embedding_path = nullptr;
  int32_t max_context_len = 0;
  uint32_t run_core_mask = 0;
  int optional_arg_start = 9;
  if (config_mode) {
    if (!load_service_config(argv[2], &service_config)) return -1;
    base_model_path = service_config.stage0_model.c_str();
    base_weight_path = service_config.stage0_weight.c_str();
    tokenizer_path = service_config.tokenizer.c_str();
    embedding_path = service_config.embedding.c_str();
    max_context_len = service_config.context_length;
    run_core_mask = service_config.core_mask;
    g_stage_count = service_config.stage_count;
    g_bucket_size = service_config.bucket_size;
    configured_device_ids = service_config.device_ids;
    optional_arg_start = 3;
  } else {
    base_model_path = argv[1];
    base_weight_path = argv[2];
    tokenizer_path = argv[3];
    embedding_path = argv[4];
    max_context_len = atoi(argv[5]);
    run_core_mask = (uint32_t)strtoul(argv[6], nullptr, 16);
    g_stage_count = (size_t)atoi(argv[7]);
    g_bucket_size = (uint64_t)strtoul(argv[8], nullptr, 10);
  }

  bool perf_mode = false;
  uint64_t perf_input_tokens = 0;
  uint64_t perf_output_tokens = 0;
  std::vector<const char*> optional_args;
  for (int i = optional_arg_start; i < argc; ++i) {
    if (strcmp(argv[i], "--perf") == 0) {
      if (perf_mode || i + 2 >= argc) {
        LOGW("--perf requires exactly one <input_tokens> <output_tokens> pair\n");
        return -1;
      }
      char* input_end = nullptr;
      char* output_end = nullptr;
      perf_input_tokens = strtoull(argv[i + 1], &input_end, 10);
      perf_output_tokens = strtoull(argv[i + 2], &output_end, 10);
      if (!input_end || *input_end != '\0' || !output_end || *output_end != '\0' ||
          perf_input_tokens == 0 || perf_output_tokens == 0) {
        LOGW("--perf token lengths must be positive integers\n");
        return -1;
      }
      perf_mode = true;
      i += 2;
      continue;
    }
    optional_args.push_back(argv[i]);
  }

  size_t optional_offset = 0;
  if (!optional_args.empty() && strcmp(optional_args[0], "--daemon") == 0) {
    g_daemon_mode = true;
    optional_offset = 1;
  }
  const char* prompt_arg = optional_args.size() > optional_offset ? optional_args[optional_offset] : nullptr;
  std::string prompt_buf;
  const char* prompt = nullptr;
  if (prompt_arg) {
    // 如果以 .txt 结尾，读取文件内容作为 prompt
    size_t arg_len = strlen(prompt_arg);
    if (arg_len >= 4 && strcmp(prompt_arg + arg_len - 4, ".txt") == 0) {
      int fd = open(prompt_arg, O_RDONLY);
      if (fd < 0) {
        printf("Failed to open prompt file: %s\n", prompt_arg);
        return -1;
      }
      struct stat st;
      if (fstat(fd, &st) != 0) {
        printf("Failed to stat prompt file: %s\n", prompt_arg);
        close(fd);
        return -1;
      }
      prompt_buf.resize(st.st_size);
      ssize_t n = read(fd, &prompt_buf[0], st.st_size);
      close(fd);
      if (n != st.st_size) {
        printf("Failed to read prompt file: %s\n", prompt_arg);
        return -1;
      }
      prompt = prompt_buf.c_str();
      printf("Loaded prompt from file: %s (%lld bytes)\n", prompt_arg, (long long)st.st_size);
    } else {
      prompt = prompt_arg;
    }
  }
  if (!prompt) {
    prompt = "system\n You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n";
  }
  int max_new_tokens = optional_args.size() > optional_offset + 1 ? atoi(optional_args[optional_offset + 1]) : 512;
  g_verbose = optional_args.size() > optional_offset + 2 ? (atoi(optional_args[optional_offset + 2]) != 0) : false;
  g_ignore_eos = optional_args.size() > optional_offset + 3 ? (atoi(optional_args[optional_offset + 3]) != 0) : false;
  const char* rope_path = optional_args.size() > optional_offset + 4 ? optional_args[optional_offset + 4] : nullptr;
  if (perf_mode) {
    // 性能测试必须跑满指定输出长度，不能被模型提前返回的 EOS 打断。
    g_ignore_eos = true;
    g_perf_mode = true;
  }

  std::vector<std::string> ext_device_ids;
  for (size_t i = optional_offset + 5; i < optional_args.size() && ext_device_ids.size() < g_stage_count; ++i) {
    ext_device_ids.push_back(optional_args[i]);
  }
  if (config_mode) {
    if (!ext_device_ids.empty()) {
      printf("device IDs must be configured in --config mode\n");
      return -1;
    }
    ext_device_ids = configured_device_ids;
  }

  if (g_stage_count < 1) {
    printf("stage_count must be >= 1, got %zu\n", g_stage_count);
    return -1;
  }
  if (max_context_len <= 0) {
    printf("max_context_len must be > 0, got %d\n", max_context_len);
    return -1;
  }

  // 工具: 将 "model_seg0.rknn" 替换为 "model_segN.rknn"
  auto replace_seg_suffix = [](const std::string& base, size_t seg_idx) -> std::string {
    // 找到最后一个 _seg 出现的位置
    size_t pos = base.rfind("_seg");
    if (pos == std::string::npos) {
      // 没有 _seg 后缀，直接在 .rknn 前插入 _segN
      size_t dot = base.rfind(".rknn");
      if (dot != std::string::npos) {
        return base.substr(0, dot) + "_seg" + std::to_string(seg_idx) + ".rknn";
      }
      return base + "_seg" + std::to_string(seg_idx);
    }
    // 有 _seg 后缀，替换其中的数字
    size_t num_start = pos + 4; // 跳过 "_seg"
    size_t num_end = num_start;
    while (num_end < base.size() && isdigit(base[num_end])) {
      ++num_end;
    }
    return base.substr(0, num_start) + std::to_string(seg_idx) + base.substr(num_end);
  };

  // 生成所有段的模型 & 权重路径
  std::vector<std::string> model_paths(g_stage_count);
  std::vector<std::string> weight_paths(g_stage_count);
  for (size_t i = 0; i < g_stage_count; ++i) {
    model_paths[i] = replace_seg_suffix(base_model_path, i);
    weight_paths[i] = replace_seg_suffix(base_weight_path, i);
  }

  // 校验所有段文件存在
  bool all_exist = true;
  for (size_t i = 0; i < g_stage_count; ++i) {
    struct stat st;
    if (stat(model_paths[i].c_str(), &st) != 0) {
      printf("ERROR: model file not found: %s\n", model_paths[i].c_str());
      all_exist = false;
    }
    if (stat(weight_paths[i].c_str(), &st) != 0) {
      printf("ERROR: weight file not found: %s\n", weight_paths[i].c_str());
      all_exist = false;
    }
  }
  if (!all_exist) {
    return -1;
  }

  PipelineState pipeline(g_stage_count);
  std::vector<StageRuntime> stages(g_stage_count);
  for (size_t i = 0; i < stages.size(); ++i) {
    stages[i].name = "stage" + std::to_string(i);
  }

  Tokenizer* tokenizer = nullptr;
  VocabInfo vocab_info;
  memset(&vocab_info, 0, sizeof(vocab_info));
  embedding_info embed_info;
  struct stat emb_st;
  memset(&emb_st, 0, sizeof(emb_st));

  if (init_tokenizer_and_embedding(tokenizer_path, embedding_path, &vocab_info, &tokenizer, &embed_info, &emb_st) != 0) {
    return -1;
  }

  g_last_stage_result.tokenizer = tokenizer;
  reset_last_stage_result();

  std::vector<int32_t> perf_tokens;
  if (perf_mode) {
    const uint64_t context_limit = (uint64_t)max_context_len;
    if (perf_input_tokens > context_limit || perf_output_tokens > context_limit - perf_input_tokens) {
      printf("invalid perf lengths: input=%llu + output=%llu must not exceed max_context_len=%llu\n",
             (unsigned long long)perf_input_tokens, (unsigned long long)perf_output_tokens,
             (unsigned long long)context_limit);
      release_resources(stages, nullptr, &embed_info, emb_st.st_size, tokenizer);
      return -1;
    }
    if (vocab_info.linefeed_id < 0 || vocab_info.linefeed_id >= vocab_info.vocab_size) {
      printf("invalid linefeed token id for perf test: %d\n", vocab_info.linefeed_id);
      release_resources(stages, nullptr, &embed_info, emb_st.st_size, tokenizer);
      return -1;
    }
    perf_tokens.assign((size_t)perf_input_tokens, vocab_info.linefeed_id);
    printf("\n=== Performance Test Mode ===\n");
    printf("Input tokens: %llu, output tokens: %llu\n",
           (unsigned long long)perf_input_tokens, (unsigned long long)perf_output_tokens);
  }

  rknn3_llm_param session_param;
  memset(&session_param, 0, sizeof(session_param));
  session_param.logits_name = (char*)"output";
  session_param.max_context_len = max_context_len;
  session_param.sampling_param.temperature = 1.0f;
  session_param.sampling_param.top_k = 1;
  session_param.sampling_param.top_p = 0.9f;
  session_param.sampling_param.repeat_penalty = 1.0f;
  session_param.sampling_param.frequency_penalty = 0.0f;
  session_param.sampling_param.presence_penalty = 0.0f;
  session_param.vocab_info.vocab_size = vocab_info.vocab_size;
  session_param.vocab_info.n_special_eos_id = vocab_info.n_special_eos_id;
  session_param.vocab_info.n_special_bos_id = vocab_info.n_special_bos_id;
  memcpy(session_param.vocab_info.special_eos_id, vocab_info.special_eos_id, sizeof(vocab_info.special_eos_id));
  memcpy(session_param.vocab_info.special_bos_id, vocab_info.special_bos_id, sizeof(vocab_info.special_bos_id));
  session_param.vocab_info.linefeed_id = vocab_info.linefeed_id;
  session_param.vocab_info.ignore_eos_token = g_ignore_eos ? 1 : 0;

  EmbedCallbackContext embed_ctx;
  embed_ctx.embed_info = &embed_info;
  embed_ctx.pipeline = &pipeline;

  InputCbUserdata input_cb_data;
  memset(&input_cb_data, 0, sizeof(input_cb_data));

  // 自动查找设备，检查设备数是否足够
  rknn3_devices devs;
  memset(&devs, 0, sizeof(devs));
  int ret = rknn3_find_devices(&devs);
  if (ret != RKNN3_SUCCESS) {
    printf("find devices failed: ret=%d\n", ret);
    release_resources(stages, &input_cb_data, &embed_info, emb_st.st_size, tokenizer);
    return -1;
  }
  printf("found %d devices:\n", devs.n_devices);
  for (int i = 0; i < devs.n_devices; ++i) {
    printf("  [%d] type=%s, id=%s\n", i, devs.devices[i].type, devs.devices[i].id);
  }

  if (ext_device_ids.empty()) {
    // 未指定外部 device_id，自动分配
    if (devs.n_devices < (int)g_stage_count) {
      printf("auto-detect failed: found=%d devices, need=%zu\n", devs.n_devices, g_stage_count);
      release_resources(stages, &input_cb_data, &embed_info, emb_st.st_size, tokenizer);
      return -1;
    }
    printf("auto-assigning %zu devices\n", g_stage_count);
  } else if ((int)ext_device_ids.size() < (int)g_stage_count) {
    printf("not enough device_id arguments: need=%zu, got=%zu\n", g_stage_count, ext_device_ids.size());
    release_resources(stages, &input_cb_data, &embed_info, emb_st.st_size, tokenizer);
    return -1;
  } else {
    // 校验外部指定的 device_id 是否在可用设备列表中
    printf("using external device_id list:\n");
    for (const auto& id : ext_device_ids) {
      printf("  %s\n", id.c_str());
    }
  }

    // 如果提供了 rope 路径，加载 rope caches 并注册 input_callback
  if (rope_path && rope_path[0] != '\0') {
    printf("loading rope cache: %s\n", rope_path);
    if (load_safetensors(rope_path, input_cb_data.rope_caches,
                         &input_cb_data.rope_fd, &input_cb_data.rope_mmap_base,
                         &input_cb_data.rope_mmap_size) != 0) {
      printf("load_safetensors failed\n");
      release_resources(stages, &input_cb_data, &embed_info, emb_st.st_size, tokenizer);
      return -1;
    }
  }

  timeval model_init_start;
  timeval model_init_end;
  gettimeofday(&model_init_start, NULL);
  bool ok = true;
  for (size_t i = 0; i < stages.size(); ++i) {
    const char* device_id = ext_device_ids.empty() ? devs.devices[i].id : ext_device_ids[i].c_str();
    ok = init_stage(stages[i], pipeline, i, device_id,
                    model_paths[i].c_str(), weight_paths[i].c_str(), session_param, run_core_mask,
                    i + 1 == stages.size(), tokenizer, &embed_ctx, &input_cb_data);
    if (!ok) {
      break;
    }
  }
  if (!ok) {
    release_resources(stages, &input_cb_data, &embed_info, emb_st.st_size, tokenizer);
    return -1;
  }
  gettimeofday(&model_init_end, NULL);
  print_model_init_statistics(stages, elapsed_us(model_init_start, model_init_end) / 1e3f);

  auto run_request = [&](const std::string& request_prompt, int request_max_new_tokens,
                         bool enable_thinking, std::string* response_text,
                         RequestMetrics* request_metrics = nullptr) -> bool {
  if (!g_daemon_mode) {
    printf("\n=== Prefill Pipeline ===\n");
  }
  timeval prefill_start;
  timeval prefill_end;
  gettimeofday(&prefill_start, NULL);
  uint64_t prefill_tokens = 0;
  const char* prefill_prompt = perf_mode ? nullptr : request_prompt.c_str();
  const std::vector<int32_t>* prefill_token_input = perf_mode ? &perf_tokens : nullptr;
  if (!run_pipeline_once(stages, pipeline, prefill_prompt, prefill_token_input, &prefill_tokens, true,
                         enable_thinking)) {
    printf("prefill failed\n");
    return false;
  }
  gettimeofday(&prefill_end, NULL);

  int next_token = -1;
  get_last_stage_token(&next_token);

  if (!g_daemon_mode) {
    printf("\n=== Decode Loop ===\n");
  }
  if (next_token < 0) {
    printf("prefill did not return token from result_callback\n");
    return false;
  }
  timeval decode_start;
  timeval decode_end;
  gettimeofday(&decode_start, NULL);
  uint64_t decode_tokens = 0;
  const uint64_t decode_limit = perf_mode ? perf_output_tokens - 1 : (uint64_t)request_max_new_tokens;
  for (uint64_t step = 0; step < decode_limit && next_token >= 0; ++step) {
    std::vector<int32_t> token_vec(1, next_token);
    VLOG("[Decode %llu] token=%d\n", (unsigned long long)(step + 1), next_token);

    if (!run_pipeline_once(stages, pipeline, nullptr, &token_vec, nullptr, false)) {
      printf("decode step %llu failed\n", (unsigned long long)(step + 1));
      break;
    }

    if (!get_last_stage_token(&next_token)) {
      printf("decode step %llu did not return token from result_callback\n", (unsigned long long)(step + 1));
      break;
    }
    decode_tokens += 1;
    if (!g_ignore_eos && next_token == vocab_info.special_eos_id[0]) {
      VLOG("\ndecode step %llu reached EOS token\n", (unsigned long long)(step + 1));
      break;
    }
  }
  gettimeofday(&decode_end, NULL);

  float prefill_ms = elapsed_us(prefill_start, prefill_end) / 1e3f;
  float decode_ms  = elapsed_us(decode_start, decode_end) / 1e3f;
  // The prefill pass emits the first generated token; each following token is
  // counted by the decode loop.  This is the completion length exposed by the
  // daemon, including an EOS token when generation stops on EOS.
  const uint64_t generated_tokens = 1 + decode_tokens;
  const float decode_tps = decode_ms > 0.0f ? 1e3f * decode_tokens / decode_ms : 0.0f;
  if (perf_mode) {
    printf("\nPerformance Test Lengths: input=%llu/%llu, output=%llu/%llu\n",
           (unsigned long long)prefill_tokens, (unsigned long long)perf_input_tokens,
           (unsigned long long)generated_tokens, (unsigned long long)perf_output_tokens);
  }
  if (g_daemon_mode) {
    // Keep metrics in the JSONL reply, but do not synchronously write a log
    // line on every request: benchmark runs should not include stderr I/O.
  } else {
    print_performance_statistics(prefill_tokens, prefill_ms, decode_tokens, decode_ms);
    print_per_stage_performance_statistics(stages);
  }
  for (size_t i = 0; i < stages.size(); ++i) {
    rknn3_session_clear_kvcache(stages[i].session, RKNN3_KVCACHE_CLEAR_ALL);
  }
  if (response_text) {
    std::lock_guard<std::mutex> lock(g_generated_text_mutex);
    *response_text = g_generated_text;
  }
  if (request_metrics) {
    request_metrics->input_tokens = prefill_tokens;
    request_metrics->output_tokens = generated_tokens;
    request_metrics->ttft_ms = prefill_ms;
    request_metrics->decode_ms = decode_ms;
    request_metrics->decode_tps = decode_tps;
  }
  return true;
  };

  if (g_daemon_mode) {
    nlohmann::json ready = {{"ready", true}, {"model", "qwen3.5-9b-110k"}, {"stage_count", g_stage_count}};
    printf("%s\n", ready.dump().c_str());
    fflush(stdout);
    std::string line;
    while (std::getline(std::cin, line)) {
      nlohmann::json reply;
      try {
        nlohmann::json request = nlohmann::json::parse(line);
        const std::string id = request.value("id", "");
        const int request_max_new_tokens = request.value("max_new_tokens", max_new_tokens);
        const bool request_enable_thinking = request.value("enable_thinking", false);
        std::string request_prompt;
        if (request.contains("messages")) {
          request_prompt = build_qwen35_chat_prompt(request.at("messages"), request_enable_thinking);
        } else if (request.contains("prompt") && request.at("prompt").is_string()) {
          // Compatibility path for SDK-level debugging only. Production
          // callers should use messages so the template stays service-owned.
          request_prompt = request.at("prompt").get<std::string>();
        } else {
          throw std::runtime_error("messages is required");
        }
        if (request_prompt.empty() || request_max_new_tokens <= 0) {
          throw std::runtime_error("prompt and positive max_new_tokens are required");
        }
        {
          std::lock_guard<std::mutex> lock(g_generated_text_mutex);
          g_generated_text.clear();
        }
        {
          std::lock_guard<std::mutex> lock(g_stream_request_id_mutex);
          g_stream_request_id = id;
        }
        {
          std::lock_guard<std::mutex> lock(g_stream_utf8_mutex);
          g_stream_utf8_pending.clear();
        }
        std::string response_text;
        RequestMetrics request_metrics;
        if (!run_request(request_prompt, request_max_new_tokens, request_enable_thinking,
                         &response_text, &request_metrics)) {
          throw std::runtime_error("multicard inference failed");
        }
        reply = {{"id", id}, {"ok", true}, {"text", response_text},
                 {"metrics", {{"input_tokens", request_metrics.input_tokens},
                              {"output_tokens", request_metrics.output_tokens},
                              {"total_tokens", request_metrics.input_tokens + request_metrics.output_tokens},
                              {"ttft_ms", request_metrics.ttft_ms},
                              {"decode_ms", request_metrics.decode_ms},
                              {"decode_tps", request_metrics.decode_tps}}}};
        {
          std::lock_guard<std::mutex> lock(g_stream_request_id_mutex);
          g_stream_request_id.clear();
        }
      } catch (const std::exception& error) {
        {
          std::lock_guard<std::mutex> lock(g_stream_request_id_mutex);
          g_stream_request_id.clear();
        }
        reply = {{"ok", false}, {"error", error.what()}};
      }
      printf("%s\n", reply.dump().c_str());
      fflush(stdout);
    }
  } else {
    std::string ignored;
    run_request(prompt, max_new_tokens, false, &ignored);
  }

  release_resources(stages, &input_cb_data, &embed_info, emb_st.st_size, tokenizer);

  return 0;
}
