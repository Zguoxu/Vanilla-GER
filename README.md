# Vanilla GER

基于 [HyPoradise / Hypo2Trans](https://github.com/Hypotheses-Paradise/Hypo2Trans) 的 VoiceBank-DEMAND 方法迁移项目。

开发方式：**Mac 编写与检查代码 → 云端小样本 GPU 验证 → 完整实验**。本地的轻量模型测试仅用于验证软件，不代表论文结果。

- 总体计划：[plan.md](plan.md)
- 新手操作教程：[docs/本地开发与Conda教程.md](docs/本地开发与Conda教程.md)
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
python scripts/prepare_voicebank.py

# 验证数据配对、提示模板、WER 和微型 LoRA 保存重载。
python -m pytest -q

# 练手：仅从训练部分取 10 条，使用 tiny.en/beam=5。
python scripts/generate_nbest.py --manifest data/audio_inputs/train.jsonl --output data/hypotheses/smoke-tiny/train.jsonl --model tiny.en --beam-size 5 --nbest 5 --device cpu --limit 10

# 只评分上面选定的 10 条；该数值不作为正式测试结果。
python scripts/evaluate.py --references data/references/train.jsonl --hypotheses data/hypotheses/smoke-tiny/train.jsonl --allow-subset --output reports/asr-smoke.json

# 按 ID 导出训练样例与不包含答案的推理输入。
python scripts/build_ger_data.py --hypotheses data/hypotheses/smoke-tiny/train.jsonl --references data/references/train.jsonl --allow-subset --output data/ger/smoke/train.json --inference-output data/ger/smoke/train-inputs.json
```

作者源码位于 `third_party/Hypo2Trans/`。外层 Git 管理固定提交和修正补丁；恢复命令是 `python scripts/fetch_upstream.py`。数据、权重、缓存和环境目录均不纳入 Git。

完整 GER 训练入口已提供，CUDA 量化与正式大模型训练需要在云端重新验证，操作说明见教程。所有正式结果在完成前都保持“待测”。
