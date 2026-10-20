#!/usr/bin/env bash
# One entry point for train/dev/test; each split has its own provenance-matched cache.
set -euo pipefail
cd "$(dirname "$0")/.."

dry_run=0
limit=""
while (($#)); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --limit)
            if (($# < 2)) || [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
                echo "--limit requires a positive integer" >&2; exit 2
            fi
            limit="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: bash scripts/prepare_full_asr.sh [--dry-run] [--limit N]"
            echo "Checks all inputs, generates separate train/dev/test candidates, and scores/exports train/dev."
            echo "--limit N uses separate smoke caches. Test scoring and GER training are separate steps."
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

runner=(conda run --no-capture-output -n vanilla-ger-asr python)
# This check runs even in dry-run mode; it does not load the model onto the GPU.
"${runner[@]}" scripts/check_asr_inputs.py
run() {
    printf '%q ' "$@"; printf '\n'
    if ((dry_run == 0)); then "$@"; fi
}

tag="whisper-large-v2-beam60"
limit_flags=()
score_flags=()
if [[ -n "$limit" ]]; then
    tag="${tag}-smoke-${limit}"
    limit_flags=(--limit "$limit")
    score_flags=(--allow-subset)
fi
for split in train dev test; do
    run "${runner[@]}" scripts/generate_nbest.py --config configs/asr-paper.json \
        --manifest "data/audio_inputs/${split}.jsonl" \
        --output "data/hypotheses/${tag}/${split}.jsonl" "${limit_flags[@]}"
done

for split in train dev; do
    run "${runner[@]}" scripts/evaluate.py --references "data/references/${split}.jsonl" \
        --hypotheses "data/hypotheses/${tag}/${split}.jsonl" "${score_flags[@]}" \
        --output "reports/${tag}-${split}.json"
    run "${runner[@]}" scripts/build_ger_data.py --hypotheses "data/hypotheses/${tag}/${split}.jsonl" \
        --references "data/references/${split}.jsonl" "${score_flags[@]}" \
        --output "data/ger/${tag}/${split}.json" --inference-output "data/ger/${tag}/${split}-inputs.json"
done
echo "Test candidates stay separate; evaluate the official test after selecting the final model using dev."
