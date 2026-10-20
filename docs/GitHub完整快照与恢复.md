# GitHub完整快照与恢复

用户选择的存放方式是：源码、配置、教程和报告正常提交到Git；整个项目目录另存为同一GitHub仓库的Release分卷快照。根目录 `.gitignore` 已删除。

普通Git保存项目自有源码及上游的固定commit、许可和补丁；实际嵌套上游仓库的源码和Git记录包含在完整快照中。只克隆源码时运行 `python scripts/fetch_upstream.py` 恢复上游；恢复完整快照时直接得到现有上游仓库。

快照发布位置：[full-snapshot-2026-10-21](https://github.com/Zguoxu/Vanilla-GER/releases/tag/full-snapshot-2026-10-21)。发布完成后的卷数、文件数、体积和验证结果见Release说明。

快照遍历 `/home/featurize/work/Vanilla-GER` 的全部目录，不应用任何Git忽略规则，包括clean/noisy原始数据及zip、Whisper权重、LoRA与checkpoint、候选、日志、缓存、上游源码，以及根仓库和上游仓库的 `.git` 记录。快照文件生成在项目目录之外，避免把备份递归装进自身。

范围是项目目录；项目外的Conda安装和Qwen用户级缓存属于外部依赖，环境快照及固定revision下载脚本已随源码保存。

GitHub普通Git文件上限为100 MiB；个人账户Git LFS单文件上限为2 GB。本项目有3个文件超过2 GB，因此完整快照以不超过1 GiB的分卷上传Release，避免遗漏这些文件。依据：[普通Git限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)、[LFS限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)。

## 下载和校验

选择项目工作盘之外、有至少40 GB空闲空间的位置。例如：

```bash
backup_dir="$HOME/Vanilla-GER-backup-20261021"
mkdir -p "$backup_dir"
gh release download full-snapshot-2026-10-21 --repo Zguoxu/Vanilla-GER --dir "$backup_dir"

# 在克隆的源码目录中验证每个分卷及归档内的每个文件。
python scripts/verify_github_backup.py "$backup_dir"
```

Release包含：

- `Vanilla-GER-full.tar.gz.part-000` 起的连续分卷。
- `FILE_MANIFEST.jsonl`：每个原始文件的路径、大小和SHA-256。
- `SHA256SUMS`：全部分卷和文件清单的SHA-256。
- `verification.json`：上传前逐文件比对的结果。
- `RESTORE.md`：恢复步骤。

## 恢复完整项目

验证通过后，解压到一个新的空目录，以保留当前工作：

```bash
restore_dir="$HOME/Vanilla-GER-restored-20261021"
mkdir -p "$restore_dir"
cat "$backup_dir"/Vanilla-GER-full.tar.gz.part-* | tar -xzf - -C "$restore_dir"
cd "$restore_dir/Vanilla-GER"
```

得到完整的原目录结构与Git记录。云端下载工具的登录凭据位于项目外，不属于项目快照；在新机器推送时使用自己的GitHub登录。
