"""Join hypotheses to labels by ID, and separately write unlabeled inference input."""
import argparse
from vanilla_ger.common import read_records, write_json
from vanilla_ger.upstream import normalize_text


def build(hypotheses, references, one_best=False, allow_subset=False):
    refs = {r["utt_id"]: r["ref_text"] for r in references}
    ids = {r["utt_id"] for r in hypotheses}
    if len(refs) != len(references) or len(ids) != len(hypotheses):
        raise ValueError("Duplicate IDs")
    if not ids <= refs.keys() or (not allow_subset and ids != refs.keys()):
        raise ValueError("Hypothesis/reference IDs do not match; --allow-subset is only for explicit smoke subsets")
    training, inference = [], []
    for row in hypotheses:
        if not row["nbest"]:
            raise ValueError(f'Empty candidates: {row["utt_id"]}')
        others = row["nbest"][1:] if not one_best else []
        item = {"utt_id": row["utt_id"], "input1": row["nbest"][0],
                "input2": "\n".join(f"{i}. {text}" for i, text in enumerate(others, 1))}
        inference.append(item)
        label = normalize_text(refs[row["utt_id"]])
        if not label:
            raise ValueError(f'Empty normalized reference: {row["utt_id"]}')
        training.append({**item, "output": label})
    return training, inference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hypotheses", required=True)
    p.add_argument("--references", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--inference-output", required=True)
    p.add_argument("--one-best", action="store_true")
    p.add_argument("--allow-subset", action="store_true")
    args = p.parse_args()
    training, inference = build(read_records(args.hypotheses), read_records(args.references), args.one_best, args.allow_subset)
    write_json(args.output, training)
    write_json(args.inference_output, inference)
    print(f"Wrote {len(training)} training records and label-free inference records")


if __name__ == "__main__":
    main()
