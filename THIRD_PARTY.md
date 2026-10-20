# 上游来源与修正方式

官方仓库：https://github.com/Hypotheses-Paradise/Hypo2Trans  
固定提交：`27824d9da316b34855157dd9f6b80ea47119e804`  
原始许可：MIT；原文见 `licenses/Hypo2Trans-MIT.txt`。其 Whisper 分支许可见 `licenses/Whisper-MIT.txt`。

`third_party/Hypo2Trans/` 是实际拉取的 Git 仓库；为避免嵌套仓库、数据与缓存误提交，外层仓库忽略该目录。外层保存固定 commit、`patches/hypo2trans.patch` 和项目代码，运行 `python scripts/fetch_upstream.py` 可以恢复作者代码并应用同一补丁。脚本遇到已有不兼容修改会停止，不会重置已有工作。

本项目 `src/vanilla_ger/ger.py` 从作者 H2T-LoRA 训练/推理实现改写，保留冻结基座、LoRA、因果语言模型训练目标及作者 H2T 模板。上游的三个入口转调项目模块，便于同一修正供本地与云端使用。

主要修正：

1. 提示模板使用相对于源码的位置，修复 1-best 占位符，允许从任意工作目录调用。
2. 基座模型、设备与量化精度可配置；CPU/MPS 本地验证使用非量化小模型，CUDA 量化留待云端验证。
3. 使用现有 PEFT k-bit API；不再写死 `cuda:2`，不再强制 `torch.compile`，适配器由 Trainer/PEFT 正常保存。
4. 支持显式 dev 文件，不再偷偷随机切分；按生成的 dev corpus WER 选择 checkpoint。
5. 默认维持完整 prompt+答案的 loss；`train_on_inputs=False` 时实际生成答案部分的 loss mask，并将该变体记入元数据。
6. 超长训练样本显式报错，不静默截断答案；推理只解码新增 token，输入文件拒绝参考标签。
7. Whisper 候选返回真实模型分数，保留全部 beam 候选后再去重取最多 5 条，不随机复制。
8. 明确添加 `large-v2` 的官方权重地址；作者旧分支只有含义不同的 `large` 条目，不能直接拿它当 large-v2。
9. 音频与文本按 ID 关联，保留候选生成的输入与模型/解码来源，评分和候选生成分开。
10. 使用 corpus WER；保留空预测的删除错误，数字不再被删除。
11. 在项目 Trainer 的单设备断点恢复入口显式使用 `torch.load(weights_only=True)`；NumPy 随机数状态仅临时许可所需类型，保留优化器、学习率和随机数的恢复行为，不修改已安装的 Transformers 文件。

规范化口径也有明确调整：作者分支把 `EnglishNumberNormalizer` 的调用注释掉了；本项目在其英文规范化之后显式调用同一源码中的数字规范化器，使 `twenty` 与 `20` 一致。候选、训练标签、参考和预测均使用同一规则，版本为 `hyporadise-english-plus-number-normalizer-v1`。它与作者生成脚本的数字转英文词后处理不是同一口径，最终报告须列明。

本地 `tiny.en/beam=5` 和随机初始化微型 LLaMA 的测试仅用于软件验收；正式论文方法迁移仍需云端 large-v2、目标 LLaMA 基座和完整训练验证。
