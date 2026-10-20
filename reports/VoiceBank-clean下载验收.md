# VoiceBank-DEMAND clean训练与测试音频下载验收

2026-10-21补充clean音频；具体检查时间见 [clean配对审计](voicebank-clean-audit.json)，时区为Asia/Shanghai。

| 文件 | 压缩包大小（字节） | 解压位置 | 音频条数 |
| --- | --- | --- | --- |
| `clean_trainset_28spk_wav.zip` | 2,486,057,279 | `data/raw/voicebank_demand/clean_trainset_28spk_wav/` | 11,572 |
| `clean_testset_wav.zip` | 154,332,064 | `data/raw/voicebank_demand/clean_testset_wav/` | 824 |

压缩包保存在 `data/raw/archives/`。28人训练部分包含既有train的10,830条与dev的742条，官方test另外824条，共12,396条clean音频。

两个压缩包都核对了官方大小、MD5和固定镜像revision的LFS SHA-256：

| 文件 | 官方MD5 |
| --- | --- |
| `clean_trainset_28spk_wav.zip` | `d2d5a45ec32f8fcbf201bde0447e20ba` |
| `clean_testset_wav.zip` | `34eb1c0ba7ef667e9b966866c542fc16` |

官方元数据来源为 [VoiceBank-DEMAND发布页](https://datashare.ed.ac.uk/items/6ed35425-bf14-4d2b-93a1-0a4984952757) 的文件API。传输使用与官方文件大小和MD5匹配的固定Hugging Face镜像。实际来源、校验值及解压大小记录在 [下载报告](downloads.json)，官方接口与镜像元数据记录在 [来源核验](voicebank-clean-source-metadata.json)。当前下载配置包含clean/noisy训练、测试及人工文本共6个zip。

`scripts/check_clean_pairs.py` 逐条核对clean与noisy：ID一一对应、均为单声道48 kHz，同一ID的采样点数一致。还检查clean的ID集合与现有train/dev/test清单一致。结果见 [配对审计](voicebank-clean-audit.json)。

补充clean后，原有三个划分的完整清单、无标签ASR输入与参考文本共9份文件均比对SHA-256，内容未变。主数据审计中的 `clean_audio_downloaded` 已更新。clean可用于后续干净语音ASR对照；含噪音频的正式候选仍使用现有输入文件。

需要在新机器重新下载或核对时，在项目根目录执行：

```bash
conda activate vanilla-ger-asr
python scripts/download_data.py --extract --files clean_trainset_28spk_wav.zip clean_testset_wav.zip --mirror-config configs/download-mirrors.json
python scripts/check_clean_pairs.py
```

本次只进行了数据补齐、校验和配对检查，未启动全量ASR或Qwen训练。
