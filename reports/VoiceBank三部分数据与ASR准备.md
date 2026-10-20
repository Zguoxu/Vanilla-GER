# VoiceBank-DEMAND三部分数据与Whisper large-v2准备

官方测试音频和人工文本已补齐、校验并解压。当前可以用同一入口依次为train、dev、test生成Whisper候选，三个划分分别保存。检查时间见 [机器验收记录](voicebank-three-split-preparation.json)，时间使用Asia/Shanghai。

后续补充：clean训练和测试音频也已下载，见 [clean下载验收](VoiceBank-clean下载验收.md)。含噪音频的三个输入清单仍按本报告使用。

## 数据已准备

| 部分 | 条数 | 音频时长 | 说话人 |
| --- | --- | --- | --- |
| train | 10,830 | 8.7472小时 | 官方训练部分中除p226/p227外的26人 |
| dev | 742 | 0.6441小时 | p226、p227 |
| test | 824 | 0.5756小时，约34.53分钟 | 官方独立测试集p232、p257 |
| 合计 | 12,396 | 9.9668小时 | 30人 |

本次补充文件来自 [官方数据发布页](https://datashare.ed.ac.uk/items/6ed35425-bf14-4d2b-93a1-0a4984952757)：

| 文件 | 大小（字节） | 官方MD5，已通过 |
| --- | --- | --- |
| `noisy_testset_wav.zip` | 170,578,854 | `fb1b86caa31e8ba5b506c0c64da9aab5` |
| `testset_txt.zip` | 440,277 | `6ca8ef3756f3769c05bb694e77ad29f0` |

音频经固定revision的镜像传输，额外核对镜像LFS SHA-256；文本直接从官方取得。传输地址、MD5和实际SHA-256见 [下载记录](downloads.json)，官方接口与镜像元数据见 [来源核验](voicebank-test-source-metadata.json)。原始zip位于 `data/raw/archives/`，解压至 `data/raw/voicebank_demand/noisy_testset_wav/` 和 `testset_txt/`。

每份测试音频都匹配同名文本。所有12,396条均为单声道48 kHz，最长15.12秒以内，适合当前不裁剪的30秒Whisper解码窗口。三部分的ID、说话人、音频哈希交叉均为0；补测试集前后的train/dev清单、ASR输入、参考文件共6份SHA-256完全一致。

参考文本存在重合：train/dev为605种、train/test为617种、dev/test为183种规范化文本。它们对应不同说话人和录音；保留官方测试划分，最终结果须说明这一数据特征。完整统计见 [数据审计](data_audit.json)。

## Whisper读取哪些文件

| 部分 | 无标签ASR输入 | 独立参考文本 | 正式候选输出 |
| --- | --- | --- | --- |
| train | `data/audio_inputs/train.jsonl` | `data/references/train.jsonl` | `data/hypotheses/whisper-large-v2-beam60/train.jsonl` |
| dev | `data/audio_inputs/dev.jsonl` | `data/references/dev.jsonl` | `data/hypotheses/whisper-large-v2-beam60/dev.jsonl` |
| test | `data/audio_inputs/test.jsonl` | `data/references/test.jsonl` | `data/hypotheses/whisper-large-v2-beam60/test.jsonl` |

Whisper输入每行只含 `utt_id`、`noisy_path`、`audio_sha256`，不含参考答案。完整审计信息位于 `data/manifests/{train,dev,test}.jsonl`。音频加载时通过FFmpeg转换为单声道16 kHz；预检查确认全部音频在30秒窗口内，生成前逐条核对音频哈希。

固定使用 `configs/asr-paper.json`：large-v2、CUDA、英语、beam=60、temperature=0、不输出时间戳；按模型分数排序并去重，保留真实的最多5条候选。现有 `.cache/whisper/large-v2.pt` 已重新核对官方SHA-256。预检查记录见 [ASR输入检查](asr-inputs-preflight.json)。

## 如何运行

在云端项目根目录执行：

```bash
cd /home/featurize/work/Vanilla-GER
# 检查清单、CUDA和权重，打印全量命令；不执行解码。
bash scripts/prepare_full_asr.sh --dry-run

# 每部分1条的实际解码检查，写入独立的smoke缓存。
bash scripts/prepare_full_asr.sh --limit 1

# 启动三部分的正式候选生成。
bash scripts/prepare_full_asr.sh
```

脚本使用 `vanilla-ger-asr` Conda环境，三部分依次运行以控制显存。每个输出都有配置与输入哈希的 `.meta.json`；重跑时复用同一来源的已有候选，配置或输入改变则停止。小样本输出位于 `data/hypotheses/whisper-large-v2-beam60-smoke-1/`，与正式缓存分开。

脚本在候选生成后自动评分train/dev，并导出它们的GER训练和无标签推理文件。test仅保留独立候选；最终方案确定后再评分，不导出为Qwen训练数据。脚本不启动Qwen训练。

## 已做检查与运行范围

数据准备的针对性测试通过：`1 passed, 11 deselected`。检查了测试集独立导出、ASR输入不带标签、原训练清单内容不变、缺失文本和跨划分音频重复会报错。Bash语法与正式命令预览也已检查。

三部分各1条的large-v2真实解码均已成功，候选数分别为4、4、5条，见 [机器验收记录](voicebank-three-split-preparation.json)。此次检查不计算官方test的WER，只确认三个输入、解码与缓存路径可用，不能当作模型效果结论。

全量12,396条候选尚未生成。依据此前3090上的10条测速，线性外推约29.4小时（含test，尚无全量实测）。本轮3条的纯解码耗时分别为13.51、7.50、34.03秒，可见单条间差异明显，原10条估算不能保证全量耗时；Qwen正式训练和最终评测另计。原先27.4小时仅针对train/dev。
