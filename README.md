# RK1828 Qwen3.5-9B 2-cards Service

面向 RK3588 + 多张 RK1828 的常驻大语言模型推理组件。它基于 Rockchip `rknn3-model-zoo/examples/multicard/cpp` 的分段流水线实现，将一个分段 `.rknn` 模型固定分配到多张 RK1828，并以 JSON Lines（JSONL）协议向上层提供请求/响应。

```text
上层应用 ── JSONL stdin/stdout ──> rk1828_qwen35_9b_2cards_daemon
                                      ├─ RK1828 #0：stage 0
                                      └─ RK1828 #1：stage 1
```

模型和 RKNN context 只在 daemon 启动时初始化一次；单个请求结束时仅释放该请求的 KV Cache，不重新加载模型。HTTP/OpenAI 兼容接口属于上层网关职责，本项目保持为可复用的原生推理内核。

## 仓库内容

| 路径 | 用途 |
| --- | --- |
| `native/src/main.cc` | 基于官方多卡示例扩展的 JSONL 常驻入口。 |
| [config/qwen35-9b.json](config/qwen35-9b.json) | 两段模型、Tokenizer、Embedding、PCIe Device ID 与上下文长度配置示例。 |
| [CMakeLists.txt](CMakeLists.txt) | 复用 Rockchip model-zoo 的 Runtime、Tokenizer 与第三方依赖的交叉编译配置。 |
| `scripts/` | 可复现构建、部署与板端验证脚本。 |
| `docs/` | JSONL 协议、模型布局与性能记录。 |

模型 `.rknn`、`.weight`、`.gguf`、`.bin` 及 Rockchip Runtime 不纳入 Git。

> 当前提交前请确认 `native/src/main.cc`、`scripts/` 和 `docs/` 中的实际实现已一并纳入仓库；`CMakeLists.txt` 依赖这些路径。仅有配置文件时不能独立完成构建。

## 模型与硬件要求

本项目不提供模型转换。需要准备同一模型的多段 RKNN 产物，以及与其匹配的权重、Tokenizer 和 Embedding 文件。以 Qwen3.5-9B 两段模型为例：

```text
Qwen3.5-9B-llm_seg0.rknn      stage 0
Qwen3.5-9B-llm_seg0.weight
Qwen3.5-9B-llm_seg1.rknn      stage 1
Qwen3.5-9B-llm_seg1.weight
Qwen3.5-9B-llm.tokenizer.gguf
Qwen3.5-9B-llm.embed.bin
```

`config/qwen35-9b.json` 中的 `device_ids` 必须与实际 PCIe 枚举一致。板端可通过以下命令确认：

```bash
rknn-smi info -l
```

示例中的 `0001:11:00.0` 与 `0003:31:00.0` 是一套已验证设备的前两张 RK1828；更换板卡或 PCIe 拓扑后必须重新配置，不能照抄。

## 快速开始

### 1. 在 x86 构建机交叉编译

构建机需要：CMake、aarch64 GNU 工具链，以及与板端 Runtime 对应版本的 `rknn3-model-zoo`。本项目不复制 SDK 的 Runtime、Tokenizer 或官方多卡实现。

```bash
cmake -S . -B build \
  -DRKNN3_MODEL_ZOO_ROOT=/path/to/rknn3-model-zoo \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64
cmake --build build -j
cmake --install build --prefix dist/RK1828-qwen3.5-9b-2cards-service
```

产物应包含：

```text
dist/RK1828-qwen3.5-9b-2cards-service/
├── bin/rk1828_qwen35_9b_2cards_daemon
└── lib/librknnrt.so
```

### 2. 部署到板端

将构建产物、配置和模型文件部署到板端。例如：

```text
/userdata/RK1828-qwen3.5-9b-2cards-service/
├── bin/rk1828_qwen35_9b_2cards_daemon
├── lib/librknnrt.so
├── config/qwen35-9b.json
└── models/qwen3.5-9b/           # 模型分段、权重、Tokenizer、Embedding
```

配置中的模型路径可以使用绝对路径，也可以由部署脚本在复制时替换为实际目录。板端 Runtime 与驱动必须和构建使用的 RKNN3 SDK 版本兼容。

### 3. 启动常驻 daemon

典型启动形式如下；实际参数以 `native/src/main.cc` 的 CLI 为准：

```bash
export LD_LIBRARY_PATH=/userdata/RK1828-qwen3.5-9b-2cards-service/lib:${LD_LIBRARY_PATH}
/userdata/RK1828-qwen3.5-9b-2cards-service/bin/rk1828_qwen35_9b_2cards_daemon \
  --config /userdata/RK1828-qwen3.5-9b-2cards-service/config/qwen35-9b.json \
  --daemon
```

启动完成后，stdout 输出一行 ready 事件：

```json
{"ready":true,"model":"qwen3.5-9b-110k","stage_count":2}
```

## JSONL 协议

每行 stdin 为一个独立请求；每行 stdout 为对应的一个 JSON 响应。上层必须根据 `id` 匹配请求与响应，且同一个 daemon 同时只接受一个请求。

请求：

```json
{
  "id": "request-001",
  "messages": [
    {"role": "system", "content": "你是简洁的中文助手。"},
    {"role": "user", "content": "用一句话解释什么是向量检索。"}
  ],
  "max_new_tokens": 128,
  "enable_thinking": false
}
```

成功响应：

```json
{"id":"request-001","ok":true,"text":"向量检索是将查询和文档编码为向量，再按相似度返回相关内容的方法。"}
```

失败响应：

```json
{"id":"request-001","ok":false,"error":"invalid request"}
```

服务内置 Qwen3.5 ChatML 模板，客户端只传 `messages`，不得自行加入 `<|im_start|>`、`<|im_end|>` 等控制 token。默认 `enable_thinking=false`：服务会闭合 assistant 的 `<think>` 区域，使模型直接输出最终答案。

## 已验证性能

测试条件：RK3588 + 3 × RK1828；Qwen3.5-9B 两段模型使用前两张卡；上下文配置为 4096；模型已完成一次性初始化。

| 项目 | 实测结果 |
| --- | ---: |
| 两卡模型初始化 wall time | 约 34 秒 |
| TTFT | 242.76 ms |
| Decode 吞吐 | 33.83 token/s |
| 请求模式 | JSONL 常驻模式，`enable_thinking=false` |

该数据仅表示模型端推理性能。上层 HTTP 网关、检索、重排、排队和网络传输时间不包含在内。

## 运行边界

- 一个 daemon 绑定其启动时配置的多张 RK1828 和分段模型；不要把同一张卡同时交给其他高负载 RKNN3 模型。
- 同一 daemon 的请求必须串行。需要多用户访问时，由上层网关排队，并在客户端断开时继续排空或显式取消当前 JSONL 请求。
- 上层如需 OpenAI 兼容的 `/v1/chat/completions`、健康检查或流式转发，应通过独立网关封装该 JSONL 内核，而不是在本项目中混入 Web 服务逻辑。
