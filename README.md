# Vanilla GER

基于 [HyPoradise / Hypo2Trans](https://github.com/Hypotheses-Paradise/Hypo2Trans) 的 VoiceBank-DEMAND 方法迁移项目。

开发方式：**Mac 编写与检查代码 → 云端小样本 GPU 验证 → 完整实验**。本地的轻量模型测试仅用于验证软件，不代表论文结果。

当前已确认主线：**Whisper large-v2 → 最多5条候选 → Qwen2.5-7B 的 8-bit 基座 + LoRA**，数据使用 VoiceBank-DEMAND。Qwen 替代了论文的第一代 LLaMA，属于方法复现与迁移实验。云端逐步操作见 [Qwen 与 VoiceBank 云端复现](docs/Qwen与VoiceBank云端复现.md)。

- 总体计划：[plan.md](plan.md)
- 新手云端操作教程：[reports/云端首次核查与操作指引.md](reports/云端首次核查与操作指引.md)
- 当前 Qwen 方案：[docs/Qwen与VoiceBank云端复现.md](docs/Qwen与VoiceBank云端复现.md)
- 目录、文件职责与阅读顺序：[docs/项目结构与代码学习指南.md](docs/项目结构与代码学习指南.md)
- 源码与修正说明：[THIRD_PARTY.md](THIRD_PARTY.md)
- 运行与数据核验结果：`reports/`

## 每次开始工作

在 Mac 终端中执行：

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate vanilla-ger-mac
cd "/Users/zguoxu/Documents/expriment/Vanilla GER"
python scripts/check_env.py
```

安装后的 Python 应指向 `~/miniforge3/envs/vanilla-ger-mac/bin/python`。结束后执行 `conda deactivate`。

## 本地阶段可执行的入口

```bash
# 只有在新机器或数据缺失时，才需要重新下载；完整文件会先校验。
python scripts/download_data.py --extract

# 生成固定 train/dev 划分与独立的无标签音频输入清单。
python scripts/prepare_voicebank.py --include-test

# 验证数据配对、提示模板、WER 和微型 LoRA 保存重载。
python -m pytest -q

# 练手：仅从训练部分取 10 条，使用 tiny.en/beam=5。
python scripts/generate_nbest.py --manifest data/audio_inputs/train.jsonl --output data/hypotheses/smoke-tiny/train.jsonl --model tiny.en --beam-size 5 --nbest 5 --device cpu --limit 10

# 只评分上面选定的 10 条；该数值不作为正式测试结果。
python scripts/evaluate.py --references data/references/train.jsonl --hypotheses data/hypotheses/smoke-tiny/train.jsonl --allow-subset --output reports/asr-smoke.json

# 按 ID 导出训练样例与不包含答案的推理输入。
python scripts/build_ger_data.py --hypotheses data/hypotheses/smoke-tiny/train.jsonl --references data/references/train.jsonl --allow-subset --output data/ger/smoke/train.json --inference-output data/ger/smoke/train-inputs.json
```

作者源码位于 `third_party/Hypo2Trans/`。外层Git管理固定提交和修正补丁，恢复命令是 `python scripts/fetch_upstream.py`；上游仓库完整副本与数据、权重、缓存一并保存到下面的Release快照。

2026-10-21完整备份方案：根目录 `.gitignore` 已删除。源码、配置、报告和教程正常提交；项目目录中的全部数据、权重、缓存、训练结果及Git记录另存为同仓库Release的分卷快照，见 [完整快照与恢复](docs/GitHub完整快照与恢复.md)。

完整 GER 训练入口已提供，CUDA 量化与正式大模型训练需要在云端重新验证，操作说明见教程。所有正式结果在完成前都保持“待测”。

## Linux 云端首次开始

先阅读 [云端首次核查与操作指引](reports/云端首次核查与操作指引.md)。其中的首次核查是带时间的环境快照；后续实际进度与验收结果另行记录。

从项目根目录创建独立 ASR 环境：

```bash
conda env create -f environment-cloud-asr.yml
conda activate vanilla-ger-asr
python scripts/check_env.py
python scripts/fetch_upstream.py
python scripts/download_data.py --extract
python scripts/prepare_voicebank.py --include-test
```

`requirements/cloud-asr.txt` 固定 Linux CUDA 版 PyTorch 及 ASR 所需依赖。`check_env.py` 将未安装的可选训练依赖记录为 `null`，并列出 `missing_packages`；该报告用于检查环境，不代表完整训练验收。GER 训练使用独立环境。

数据准备通过后，先在 GPU 上检查 10 条训练音频，使用独立的云端缓存：

```bash
python scripts/generate_nbest.py --manifest data/audio_inputs/train.jsonl --output data/hypotheses/cloud-smoke-tiny/train.jsonl --model tiny.en --beam-size 5 --nbest 5 --device cuda --limit 10
python scripts/evaluate.py --references data/references/train.jsonl --hypotheses data/hypotheses/cloud-smoke-tiny/train.jsonl --allow-subset --output reports/asr-cloud-smoke.json
```

此处是小样本软件与 GPU 检查。正式实验按 `plan.md` 使用 large-v2 及固定解码配置，并固定已选 Qwen 基座的来源、revision 与 tokenizer。

当前 GER 基座已按用户选择改为 `Qwen/Qwen2.5-7B`。训练环境使用 `environment-cloud-train.yml`；先按当前 Qwen 教程准备32条 train、4条 dev 的练手文件，再执行：

```bash
conda activate vanilla-ger-train
python scripts/train_ger.py --config configs/ger-qwen2.5-7b-smoke.json
```

正式训练的候选配置位于 `configs/ger-qwen2.5-7b.json`，须在数据、长度和资源验收后使用。

官方测试音频和人工文本现已补齐；train/dev/test分别为10,830/742/824条。用下面的入口检查三部分输入，并预览统一的large-v2/beam=60候选生成命令：

```bash
bash scripts/prepare_full_asr.sh --dry-run
# 每部分1条的入口检查；使用独立缓存。
bash scripts/prepare_full_asr.sh --limit 1
# 全量候选生成：预计需要长时间GPU运行。
bash scripts/prepare_full_asr.sh
```

三部分依次生成独立候选，不混入同一训练集；test最终评分单独进行。具体文件与验收见 [三部分数据准备报告](reports/VoiceBank三部分数据与ASR准备.md)。

clean训练和测试音频已另外补齐，配对检查入口为 `python scripts/check_clean_pairs.py`，结果见 [clean下载验收](reports/VoiceBank-clean下载验收.md)。
