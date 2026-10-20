"""Corpus WER with edit counts, including empty predictions as deletions."""


def word_errors(reference, hypothesis):
    ref, hyp = reference.split(), hypothesis.split()
    if not ref:
        raise ValueError("Empty reference after normalization; inspect the label.")
    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j
    for i, a in enumerate(ref, 1):
        for j, b in enumerate(hyp, 1):
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + (a != b))
    i, j = len(ref), len(hyp)
    substitutions = deletions = insertions = 0
    while i or j:
        if i and j and ref[i-1] == hyp[j-1] and dp[i][j] == dp[i-1][j-1]:
            i, j = i-1, j-1
        elif i and j and dp[i][j] == dp[i-1][j-1] + 1:
            substitutions += 1
            i, j = i-1, j-1
        elif i and dp[i][j] == dp[i-1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    return {"substitutions": substitutions, "deletions": deletions, "insertions": insertions,
            "errors": dp[-1][-1], "reference_words": len(ref)}


def corpus_wer(pairs):
    totals = dict(substitutions=0, deletions=0, insertions=0, errors=0, reference_words=0)
    count = 0
    for reference, hypothesis in pairs:
        for key, value in word_errors(reference, hypothesis).items():
            totals[key] += value
        count += 1
    if not count:
        raise ValueError("No utterances to score.")
    return {**totals, "utterances": count, "wer_percent": totals["errors"] / totals["reference_words"] * 100}


def score_predictions(references, predictions):
    refs = {row["utt_id"]: row["ref_text"] for row in references}
    preds = {row["utt_id"]: row["prediction"] for row in predictions}
    if len(refs) != len(references) or len(preds) != len(predictions):
        raise ValueError("Duplicate utterance IDs.")
    if refs.keys() != preds.keys():
        raise ValueError(f"Prediction/reference IDs differ: missing={len(refs.keys()-preds.keys())}, extra={len(preds.keys()-refs.keys())}")
    from .upstream import normalize_text
    details = []
    for uid, ref in refs.items():
        ref, hyp = normalize_text(ref), normalize_text(preds[uid])
        details.append({"utt_id": uid, "reference": ref, "prediction": hyp, **word_errors(ref, hyp)})
    result = corpus_wer((r["reference"], r["prediction"]) for r in details)
    return result, details
