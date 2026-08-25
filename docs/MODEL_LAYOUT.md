# 模型与两卡布局

配置文件只指定 `stage0_model` 与 `stage0_weight`。服务自动将路径中的 `_seg0` 替换为 `_seg1`，从而加载第二段模型和权重。

```text
stage0_model:  Qwen3.5-9B-llm_seg0.rknn
自动派生:      Qwen3.5-9B-llm_seg1.rknn

stage0_weight: Qwen3.5-9B-llm_seg0.weight
自动派生:      Qwen3.5-9B-llm_seg1.weight
```

配置的 `device_ids` 必须与 `stage_count` 等长：第 0 个 ID 对应 stage 0，第 1 个 ID 对应 stage 1。

```bash
rknn-smi info -l
```

示例配置中的 PCIe ID 仅适用于已验证设备；任何板卡、插槽或驱动变化后，都应重新确认实际 ID。

本项目不包含模型、权重、Tokenizer 或 Embedding 文件。它们应放在板端持久化目录，且配置中的绝对路径必须可读。
