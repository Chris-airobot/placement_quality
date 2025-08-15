import json
import os
from tqdm import tqdm


def replace_scores_with_indices(
    experiment_results_path: str,
    predictions_path: str,
    output_path: str | None = None,
    one_based: bool = True,
):
    """
    Replace each record's 'prediction_score' in experiment results JSONL with the
    value looked up by index from the predictions file.

    Index semantics:
    - If an experiment record has field 'index' == k, we use the k-th value from
      the predictions file (1-based if one_based=True) and set it as prediction_score.
    - If 'index' is missing, we fall back to the record's line order.

    - experiment_results_path: JSONL with one JSON object per line, containing 'prediction_score'.
    - predictions_path: JSON or JSONL with one JSON object per line, each holding a score (key: 'pred_no_collision' preferred).
    - output_path: path to write the modified JSONL. If None, writes alongside input with suffix '.indexed.jsonl'.
    - one_based: if True, indices start at 1; else 0.
    """

    if output_path is None:
        root, ext = os.path.splitext(experiment_results_path)
        output_path = f"{root}.indexed.jsonl"

    # Load prediction scores into a list for O(1) index lookup
    pred_scores: list[float] = []
    with open(predictions_path, "r") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                # Stop if predictions file is a single large JSON array; attempt to load once
                f.seek(0)
                try:
                    arr = json.load(f)
                    for it in arr:
                        v = (
                            it.get("pred_no_collision")
                            if isinstance(it, dict)
                            else None
                        )
                        if v is not None:
                            pred_scores.append(float(v))
                except Exception:
                    pass
                break
            else:
                # Typical JSONL line
                v = obj.get("pred_no_collision")
                if v is None:
                    v = obj.get("prediction_score")
                if v is None:
                    v = obj.get("score")
                if v is not None:
                    pred_scores.append(float(v))

    total = len(pred_scores)
    print(f"Loaded {total} prediction scores from {predictions_path}")

    # Process experiment results line by line
    replaced = 0
    with open(experiment_results_path, "r") as fin, open(output_path, "w") as fout:
        for i, line in enumerate(tqdm(fin, desc="Replacing scores")):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Write through unmodified if malformed
                fout.write(line)
                continue

            # Determine index for lookup: prefer explicit 'index' field
            if "index" in rec and isinstance(rec["index"], int):
                k = rec["index"]
                idx = (k - 1) if one_based else k
            else:
                # Fallback to file order
                idx = (i if one_based else i - 1)

            # Replace if within range
            if 0 <= idx < total and "prediction_score" in rec:
                rec["prediction_score"] = pred_scores[idx]
                replaced += 1

            # Write updated record
            fout.write(json.dumps(rec) + "\n")

    print(f"Updated 'prediction_score' in {replaced} records → {output_path}")


if __name__ == "__main__":
    # Defaults to the paths you mentioned; adjust as needed
    experiment_results_path = \
        "/home/chris/Chris/placement_ws/src/data/box_simulation/v5/experiments/experiment_results_test_data.jsonl"
    predictions_path = \
        "/home/chris/Chris/placement_ws/src/placement_quality/cube_generalization/test_data_predictions_zero_emb.json"
    output_path = \
        "/home/chris/Chris/placement_ws/src/data/box_simulation/v5/experiments/experiment_results_test_data_zero_emb.jsonl"

    replace_scores_with_indices(
        experiment_results_path=experiment_results_path,
        predictions_path=predictions_path,
        output_path=output_path,
        one_based=True,
    )


