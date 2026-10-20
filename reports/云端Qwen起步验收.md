# 云端 Qwen 与 VoiceBank-DEMAND 起步验收

日期：2026-10-20。项目位于 `/home/featurize/work/Vanilla-GER`。本报告记录已实际执行的小样本检查；全量实验尚未开始。

**后续更新：** 官方test音频与文本现已补齐，独立测试清单824条。train/dev/test的large-v2统一入口也已准备，三部分各1条解码通过，见 [三部分数据与ASR准备报告](VoiceBank三部分数据与ASR准备.md)。下文27.4小时是起步时只含train/dev的估算。

## 已完成的准备

- 实际设备为 **RTX 3090 24 GB**，6个逻辑 CPU。ASR 与 GER 分进程运行。
- 独立环境 `vanilla-ger-asr` 和 `vanilla-ger-train` 均为 Python 3.10.22、PyTorch 2.5.1+cu121。训练使用 Transformers 4.46.3、PEFT 0.13.2、bitsandbytes 0.44.1。`pip check` 通过，当前 bitsandbytes 的 INT8、NF4 CUDA 内核检查均通过。
- VoiceBank-DEMAND 的训练含噪音频和文本已经下载、校验并解压：11,572条，train 10,830条、dev 742条，dev说话人为p226/p227。单声道48 kHz，音频哈希无交叉；有605种规范化参考文本在两部分重复，如实保留。官方test以及clean训练/测试音频已在后续步骤补齐，clean详情见 [clean下载验收](VoiceBank-clean下载验收.md)。
- Whisper large-v2 权重已核对官方 SHA-256；Qwen 四个 safetensors 分片已逐一核对 Hub LFS SHA-256。

配置、依赖和检查证据：

- [训练环境](environment-linux-train.json)、[量化内核检查](bitsandbytes-cuda-check.json)、[完整依赖快照](../requirements/cloud-train.lock.txt)。
- [数据审计](data_audit.json)、[下载校验](downloads.json)。原 Mac 报告保存在 `history-macos/`，不可当作当前云端检查结果。
- [Qwen 权重校验](qwen-model-download.json)、[Whisper 权重校验](whisper-large-v2-download.json)。

## 真实7B训练闭环

使用 `Qwen/Qwen2.5-7B` 基础版，固定 revision `d149729398750b98c0af14eb82c78cfe92750796`。原论文使用第一代 LLaMA，本实验明确记录基座与规模迁移。

练手候选由 tiny.en/beam=5 生成，与正式 large-v2 缓存分开。32条 train、4条 dev，8-bit 基座、LoRA r=8，有效 batch=4，共8次参数更新。只训练5,046,272个 LoRA 参数；损失和梯度有限，适配器非零，新进程重载后的原始预测与训练内验证预测完全一致。

| 检查项 | 实测 |
| --- | --- |
| Trainer 运行时间 | 51.50秒；外围训练计时63.70秒，不含独立重载推理 |
| PyTorch 峰值分配显存 | 19.02 GiB |
| PyTorch 峰值保留显存 | 20.11 GiB |
| tiny.en 基线 dev WER | 32.65%（16/49词错误） |
| Qwen 练手 dev WER | 32.65%（16/49词错误） |

这4条 dev 上没有 WER 改善，不能据此判断完整方法效果。练手候选中一条出现重复生成，完整训练样本长1212 tokens，因此练手 cutoff=1280；正式 cutoff=512 尚需正式候选的长度审计，不会静默截断答案。

训练结果在 `outputs/cloud-smoke-qwen2.5-7b-int8/`。证据见 [训练与重载检查](qwen-smoke-verification.json)、[独立评分](ger-cloud-smoke-qwen-dev.json)、[token审计](qwen-smoke-token-audit.json)。

实测还修正了两个兼容问题：bitsandbytes 0.45.5 与当前 PEFT 的 INT8 LoRA API 不兼容，已固定为0.44.1；新进程量化推理对齐训练时非量化参数的 FP32 dtype，修复重载输出差异。失败日志和修复前结果保留在本地历史目录。

## 正式 ASR 配置测速

固定 Whisper large-v2、beam=60、按分数去重后最多5条候选。在 dev 的前10条（p226_001至p226_010）完成解码及独立评分，真实候选不足5条时不复制补齐。

| 检查项 | 实测 |
| --- | --- |
| 优化后解码时间 | 85.31秒，共10条；8.53秒/条 |
| 模型加载时间 | 26.19秒，单独记录 |
| PyTorch 峰值分配/保留显存 | 20.59 / 23.07 GiB |
| 1-best WER | 6.38%（9/141词错误） |
| N-best oracle WER | 3.55%（5/141词错误） |

上游交叉注意力原先在缓存命中时仍重复计算音频投影。修复后，该10条由17.60降至8.53秒/条，快2.06倍。10条 large-v2 和36条 tiny.en 的候选、顺序与分数均与修复前完全一致。见 [缓存优化对照](whisper-cache-optimization.json) 和 [10条评分](asr-large-v2-beam60-probe-10.json)。这是实现加速，未降低 beam 或更换 ASR 模型。

按这10条速度线性外推，训练加验证共11,572条的候选生成约需 **27.4小时（3090）**。样本来自单个说话人的前10条，长度和生成情况可能不代表全量；这不是已运行全量得到的时间。

4090 尚未实测。以下只作条件换算，不能当成硬件基准：

| 4090 相对当前3090的实际加速比例 | 同一批11,572条 ASR 的估算 |
| --- | --- |
| 1.2倍 | 22.9小时 |
| 1.5倍 | 18.3小时 |
| 2倍 | 13.7小时 |

4090与3090均为24 GB显存，见 [NVIDIA 4090规格](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/) 与 [NVIDIA 3090规格](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/)。实际速度还受 CPU 上的 beam 搜索、音频读取和逐步生成影响；不能只用理论计算性能比例推断总耗时。更换实例后先用相同10条与配置复测。

上述耗时均**不包含 Qwen 正式训练、正式 dev 生成评测和官方 test**。练手数据量、候选模型、长度上限、有效 batch 与验证 beam 均不同，不能将练手51.50秒直接按数据量放大来承诺正式训练时间。

## 软件检查与后续步骤

`python -m pytest -q -W error::FutureWarning`：**15 passed, 6 warnings in 66.20s**。覆盖微型 LLaMA/Qwen2、CUDA INT8 LoRA 参数更新、新进程预测一致性、缓存投影复用和断点恢复。6条 UserWarning 是 INT8 算子将 FP32 输入转为 FP16 的明确提示；FutureWarning 仍按错误处理。见 [测试记录](cloud-software-tests.json)。

接下来按顺序完成：

1. 决定计算时长预算或更换实例后，按正式配置生成全量 train/dev 候选；可用 `scripts/prepare_full_asr.sh`，支持复用匹配来源的缓存。
2. 审计正式 prompt 长度、全量 baseline/oracle WER，用正式配置量测 Qwen 训练和验证速度。
3. 在固定划分上正式训练，以dev选择checkpoint；官方test现已独立准备，冻结方案后再进行最终评分。

操作命令见 [云端复现指南](../docs/Qwen与VoiceBank云端复现.md)。初始训练时的源码、配置与报告哈希见 [起步实验来源记录](cloud-experiment-provenance.json)，该历史快照早于测试集补齐；当前三部分数据状态见上方后续报告。
