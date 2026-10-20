import argparse
from vanilla_ger.common import read_records, write_json, write_jsonl
from vanilla_ger.metrics import score_predictions, corpus_wer, word_errors
from vanilla_ger.upstream import normalize_text, NORMALIZATION_VERSION


def main():
    p = argparse.ArgumentParser(description="Corpus WER; references are loaded only by the scorer")
    p.add_argument("--references", required=True)
    p.add_argument("--predictions", help="JSONL with utt_id and prediction")
    p.add_argument("--hypotheses", help="Instead score ASR 1-best and N-best oracle")
    p.add_argument("--allow-subset", action="store_true", help="Explicit smoke subsets only")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if bool(args.predictions) == bool(args.hypotheses):
        p.error("Choose exactly one of --predictions or --hypotheses")
    refs = read_records(args.references)
    rows = read_records(args.predictions or args.hypotheses)
    if args.allow_subset:
        ids = {r["utt_id"] for r in rows}
        refs = [r for r in refs if r["utt_id"] in ids]
    if args.predictions:
        result, details = score_predictions(refs, rows)
    else:
        predictions = [{"utt_id": r["utt_id"], "prediction": r["nbest"][0]} for r in rows]
        result, details = score_predictions(refs, predictions)
        lookup = {r["utt_id"]: normalize_text(r["ref_text"]) for r in refs}
        oracle = []
        for row in rows:
            ref = lookup[row["utt_id"]]
            candidates = [normalize_text(c) for c in row["nbest"]]
            best = min(candidates, key=lambda c: word_errors(ref, c)["errors"])
            oracle.append((ref, best))
        result = {"baseline": result, "nbest_oracle": corpus_wer(oracle),
                  "fewer_than_five": sum(len(r["nbest"]) < 5 for r in rows)}
    result["normalization_version"] = NORMALIZATION_VERSION
    write_json(args.output, result)
    write_jsonl(args.output + ".details.jsonl", details)
    print(result)


if __name__ == "__main__":
    main()
