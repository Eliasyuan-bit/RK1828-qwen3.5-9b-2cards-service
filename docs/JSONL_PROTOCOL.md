# JSONL 协议

daemon 从标准输入逐行读取 JSON 请求，并逐行写入 JSON 响应。一个 daemon 只处理一个活动请求；上层网关如有并发访问必须排队。

## 启动

```bash
rk1828_qwen35_9b_2cards_daemon --config qwen35-9b.json --daemon
```

启动成功后先输出：

```json
{"ready":true,"model":"qwen3.5-9b-110k","stage_count":2}
```

## 请求

```json
{"id":"req-1","messages":[{"role":"user","content":"你好"}],"max_new_tokens":128,"enable_thinking":false}
```

`messages` 使用服务内置 Qwen3.5 ChatML 模板；不要在 `content` 中加入控制 token。`enable_thinking=false` 时，服务直接生成最终答案。

## 流式事件

生成期间可能输出零到多条：

```json
{"id":"req-1","event":"delta","text":"你好"}
```

`text` 是 UTF-8 完整片段。最终响应中的 `text` 为完整输出，因此上层应将 delta 累积后用最终响应校正。

## 最终响应

```json
{"id":"req-1","ok":true,"text":"你好，有什么可以帮助你？","metrics":{"input_tokens":12,"output_tokens":9,"total_tokens":21,"ttft_ms":243.1,"decode_ms":237.2,"decode_tps":33.7}}
```

失败时返回：

```json
{"ok":false,"error":"invalid request"}
```
