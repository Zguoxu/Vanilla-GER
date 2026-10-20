# Qwen2.5-7B 与 VoiceBank-DEMAND 云端复现

方案确认日期：2026-10-20。这里使用 HyPoradise 的 H2T-LoRA 方法，迁移到 VoiceBank-DEMAND，并将语言模型基座替换为 Qwen2.5-7B。实际验收结果见 `reports/`；软件测试、小样本结果和正式结果分别记录。

当前已完成下面的小样本步骤：32条 train、4条 dev 的真实 Qwen 7B INT8 LoRA 训练与新进程重载，以及 large-v2/beam=60 的10条测速。结果见 [云端起步验收](../reports/云端Qwen起步验收.md)。官方 test 的音频和文本现已补齐；全量候选、正式 GER 训练和官方 test 评测尚未完成。

## 先理解模型分工

```text
含噪音频 → 冻结的 Whisper → 候选文字 → Qwen + LoRA → 纠正后的文字
人工文本 → 训练监督 / 独立评分
```

Whisper 负责听声音，Qwen 负责读候选并纠错。LoRA 只训练基座上新增的少量参数。人工文本是训练答案和评分依据；纠错推理使用的输入文件不含答案。

| 设置 | 当前选择 |
| --- | --- |
| 正式 ASR | Whisper large-v2，beam=60，按分数去重后保留最多5条候选 |
| 练手 ASR | tiny.en，beam=5；使用独立缓存 |
| GER 基座 | `Qwen/Qwen2.5-7B` 基础版 |
| 模型 revision | `d149729398750b98c0af14eb82c78cfe92750796` |
| 基座量化 | 主线先验收 8-bit；4-bit 单独标记 |
| LoRA | r=8，alpha=16，dropout=0.05；q/k/v/o 投影 |
| 学习率 | 1e-4 |
| 训练目标 | 保留作者的完整 prompt + 答案损失 |
| 正式有效 batch | 128；单卡 micro batch=1、梯度累积128次 |
| 练手训练 | 32条 train、4条 dev；有效 batch=4，先检查8次参数更新 |
| 练手长度上限 | 1280；实际最长1212 tokens，包含一条 tiny.en 的重复候选坏例 |

Qwen 使用自己的 tokenizer，并保留作者 H2T 提示模板。训练和重载推理使用相同的 tokenizer、基座 revision、模板及量化设置。

## 第一步：理解两套环境

已配置的环境分别是：

- `vanilla-ger-asr`：解码音频、Whisper 候选生成。
- `vanilla-ger-train`：Qwen、量化、LoRA、软件测试和纠错推理。

进入云端终端时：

```bash
cd /home/featurize/work/Vanilla-GER
source /environment/miniconda3/etc/profile.d/conda.sh
conda activate vanilla-ger-train
python -c "import sys; print(sys.executable)"
python -m pip check
```

预期 Python 路径包含 `envs/vanilla-ger-train/`。运行 ASR 命令前切换为 `conda activate vanilla-ger-asr`。环境切换改变 Python 和库，项目路径仍是同一个。

在新机器重建时，从项目根目录分别执行 `conda env create -f environment-cloud-asr.yml` 和 `conda env create -f environment-cloud-train.yml`。实际版本快照在 `requirements/cloud-*.lock.txt` 与 `requirements/conda-linux-64-*.explicit.txt`。

训练环境固定 `peft==0.13.2`、`bitsandbytes==0.44.1`。实测0.45.5的量化内核能运行，但该 PEFT 版本在创建 INT8 LoRA 时会访问其已移除的属性；因此不能仅凭“量化加载成功”就判定训练兼容。当前版本已通过真实训练和重载检查。

## 第二步：检查数据和权重

数据下载、校验与解压：

```bash
python scripts/download_data.py --extract --mirror-config configs/download-mirrors.json
python scripts/prepare_voicebank.py --include-test
```

镜像提供与官方原始 zip 相同的文件；继续保留官方大小、MD5 和固定 SHA-256 检查。传输来源写入 `reports/downloads.json`，原 Mac 报告已保存在 `reports/history-macos/`。

已核实：官方训练部分11,572条，固定为train 10,830条、dev 742条（p226/p227）；官方test为824条（p232/p257）。合计12,396条，三部分ID、说话人、音频哈希均无交叉。以 `reports/data_audit.json` 为准。下载配置现包含训练和测试的clean/noisy音频及人工文本共6个zip，clean补充结果见 [clean下载验收](../reports/VoiceBank-clean下载验收.md)。

只补下载测试文件时可执行：

```bash
python scripts/download_data.py --extract --files noisy_testset_wav.zip testset_txt.zip --mirror-config configs/download-mirrors.json
python scripts/prepare_voicebank.py --include-test
```

准备测试集后，原有train/dev的清单、ASR输入和参考文件共6份已比对SHA-256，内容完全未变。测试集有独立的 `data/manifests/test.jsonl`、`data/audio_inputs/test.jsonl` 和 `data/references/test.jsonl`。

只补下载clean音频并检查与noisy的对应关系时执行：

```bash
python scripts/download_data.py --extract --files clean_trainset_28spk_wav.zip clean_testset_wav.zip --mirror-config configs/download-mirrors.json
python scripts/check_clean_pairs.py
```

clean分别位于 `data/raw/voicebank_demand/clean_trainset_28spk_wav/` 和 `clean_testset_wav/`。配对报告为 `reports/voicebank-clean-audit.json`，可用于后续干净语音ASR对照；当前正式候选入口读取含噪音频。

权重准备：

```bash
python scripts/download_ger_model.py --config configs/ger-qwen2.5-7b-smoke.json
python scripts/download_asr_model.py --model large-v2
```

Qwen 权重存入当前用户的 Hugging Face 缓存，不占30 GiB工作盘的模型空间。四个 safetensors 分片逐一对照 Hub 文件元数据检查 SHA-256，记录在 `reports/qwen-model-download.json`。Whisper 权重存入项目 `.cache/whisper/`，对照官方权重 URL 中的 SHA-256 校验。

## 第三步：练手候选与训练文件

先激活 ASR 环境，生成32条 train 和4条 dev 候选：

```bash
conda activate vanilla-ger-asr
python scripts/generate_nbest.py --manifest data/audio_inputs/train.jsonl --output data/hypotheses/cloud-smoke-tiny-32/train.jsonl --model tiny.en --beam-size 5 --nbest 5 --device cuda --limit 32
python scripts/generate_nbest.py --manifest data/audio_inputs/dev.jsonl --output data/hypotheses/cloud-smoke-tiny-32/dev.jsonl --model tiny.en --beam-size 5 --nbest 5 --device cuda --limit 4
python scripts/evaluate.py --references data/references/dev.jsonl --hypotheses data/hypotheses/cloud-smoke-tiny-32/dev.jsonl --allow-subset --output reports/asr-cloud-smoke-tiny-dev.json
python scripts/build_ger_data.py --hypotheses data/hypotheses/cloud-smoke-tiny-32/train.jsonl --references data/references/train.jsonl --allow-subset --output data/ger/cloud-smoke-tiny-32/train.json --inference-output data/ger/cloud-smoke-tiny-32/train-inputs.json
python scripts/build_ger_data.py --hypotheses data/hypotheses/cloud-smoke-tiny-32/dev.jsonl --references data/references/dev.jsonl --allow-subset --output data/ger/cloud-smoke-tiny-32/dev.json --inference-output data/ger/cloud-smoke-tiny-32/dev-inputs.json
```

验收：每条候选有稳定 ID 和来源；真实候选不足5条时保留实际条数。带 `output` 的文件用于监督训练，不带答案的 `*-inputs.json` 用于推理。`--allow-subset` 表示此处明确只用少量样本。

本次真实 token 审计发现 `p228_003` 的 tiny.en 候选出现较长重复，完整 prompt+答案共1212 tokens。因此仅将练手配置上限设为1280，保留这条样本与原始候选。正式配置的512是待正式候选审计的起点。

## 第四步：训练、保存、退出和重载

```bash
conda activate vanilla-ger-train
python scripts/train_ger.py --config configs/ger-qwen2.5-7b-smoke.json
python scripts/infer_ger.py --checkpoint outputs/cloud-smoke-qwen2.5-7b-int8 --inputs data/ger/cloud-smoke-tiny-32/dev-inputs.json --output outputs/cloud-smoke-qwen2.5-7b-int8/dev-predictions.jsonl --device cuda
python scripts/evaluate.py --references data/references/dev.jsonl --predictions outputs/cloud-smoke-qwen2.5-7b-int8/dev-predictions.jsonl --allow-subset --output reports/ger-cloud-smoke-qwen-dev.json
```

验收：只有 LoRA 参数可训练；损失和梯度有限；适配器已保存且非零；新进程能加载并输出预测。训练元数据会保存训练参数量、耗时及 PyTorch 峰值显存。少量数据上的 WER 用于排错，改善不作为正式结论。

## 第五步：验证正式 Whisper 设置

在 ASR 环境先测固定 dev 的10条：

```bash
conda activate vanilla-ger-asr
python scripts/generate_nbest.py --config configs/asr-paper.json --manifest data/audio_inputs/dev.jsonl --output data/hypotheses/large-v2-beam60-probe-10/dev.jsonl --limit 10
python scripts/evaluate.py --references data/references/dev.jsonl --hypotheses data/hypotheses/large-v2-beam60-probe-10/dev.jsonl --allow-subset --output reports/asr-large-v2-beam60-probe-10.json
```

先检查峰值显存、秒/条、baseline WER 和 oracle WER，再估算完整候选生成所需时间。每个不同模型、beam和样本集合使用自己的缓存目录。

完整候选、长度审计和资源检查通过后，再使用 `configs/ger-qwen2.5-7b.json` 训练正式迁移基线。模型选择来自固定dev；官方test保持独立，方案冻结后统一评分。

## 第六步：一个入口准备train、dev、test的正式候选

“三部分一起输入”使用同一套large-v2/beam=60配置，依次处理三个输入文件，每部分有独立输出。入口只接收ID、音频路径和哈希；音频由FFmpeg重采样为16 kHz，人工转写不传给Whisper。

先检查所有清单、GPU和权重SHA-256，并预览正式命令：

```bash
bash scripts/prepare_full_asr.sh --dry-run
```

小样本入口验证（每部分只取1条，写入独立的 `whisper-large-v2-beam60-smoke-1/` 缓存）：

```bash
bash scripts/prepare_full_asr.sh --limit 1
```

确定计算时长预算后，启动全量候选：

```bash
bash scripts/prepare_full_asr.sh
```

正式输入与输出一一对应：

| 数据部分 | Whisper输入 | 候选输出 |
| --- | --- | --- |
| train | `data/audio_inputs/train.jsonl` | `data/hypotheses/whisper-large-v2-beam60/train.jsonl` |
| dev | `data/audio_inputs/dev.jsonl` | `data/hypotheses/whisper-large-v2-beam60/dev.jsonl` |
| test | `data/audio_inputs/test.jsonl` | `data/hypotheses/whisper-large-v2-beam60/test.jsonl` |

脚本支持复用来源一致的候选缓存；配置或输入不匹配时会停止。生成后自动评分和导出train/dev的GER数据；test保留独立候选，不自动评分或导出为训练文件。脚本不会启动Qwen训练。检查报告为 `reports/asr-inputs-preflight.json`。

3090此前10条解码实测外推：train/dev约27.4小时，加入824条test后合计约29.4小时。样本量小，这只是线性估算，尚未运行全量；4090需在同一配置上重新测速。Qwen正式训练和最终评测时间另计。
