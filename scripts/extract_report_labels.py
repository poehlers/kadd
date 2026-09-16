#!/usr/bin/env python3
"""Extract the 12 knee-abnormality labels from radiology report text for the
4,349 training studies that have a report but no direct label — using a
local, open-source LLM via Ollama.

This is an OFFLINE, ONE-TIME precompute step — not something called during
training or inference (see CLAUDE.md / conversation record for the full
reasoning: redundant recomputation across 100 epochs, reproducibility of a
fixed label set, and the fact that nothing about this needs Kaggle's GPU or
environment — it's pure local text processing against the Report column
already in data/train.csv). Output is a static CSV consumed by
kadd/classify/data.py like any other label file, clearly named as
LLM-derived ("silver") rather than radiologist-verified ("gold").

SWITCHED FROM CLAUDE OPUS 5 TO LOCAL OLLAMA (2026-08-14): the Anthropic
account had $0 credit balance and the user chose to avoid further cost
rather than add billing. Model: qwen2.5:7b (strong open multilingual
coverage, sized to fit this machine's 16GB RAM for a run that will take
hours). Real timing test on one gold-labeled report: 17.6s (792 prompt
tokens, 109 output tokens) — extrapolated across all 4,407 studies, this is
a ~21+ hour run. That's the real cost of going local instead of paying for
Opus 5's Batch API: free, but slow. Structured output still works the same
way — Ollama's `format` field accepts a JSON schema exactly like Anthropic's
`output_config.format` (verified empirically against Ollama 0.32.4), so the
label schema and system prompt are unchanged from the Claude version.

No batch-job concept exists for local inference (Ollama is a synchronous
local server, not an async job queue), so this doesn't have the
submit/status/collect shape the Claude version had. Instead: a single `run`
command that processes reports sequentially, checkpointing to disk after
every report so it can be safely interrupted (Ctrl-C, laptop sleep, crash)
and resumed without redoing work or losing partial progress.

Same known limitation as before, still accepted as an explicit scope
decision: zero of the 58 gold-labeled studies are all-negative (see
CLAUDE.md §8) — the gold set is curated toward pathology-rich cases, so
calibration here says nothing about whether the model correctly recognizes
a genuinely normal report. Proceeding on the same assumption as before:
trust the 58 gold labels as-is (never touched by this script), assume
non-gold studies are also abnormal cases for now.

Usage:
    python scripts/extract_report_labels.py dry-run [--limit N]
    python scripts/extract_report_labels.py run [--limit N] [--model qwen2.5:7b]
    python scripts/extract_report_labels.py run --resume     # continues after interruption
    python scripts/extract_report_labels.py finalize          # write silver labels + calibration
"""
import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import pandas as pd
import requests

# stdout is fully buffered (not line-buffered) once redirected to a file/pipe
# rather than a TTY — without this, progress prints sit invisible in the
# buffer for a long time during a run that's meant to be tailed live for
# hours. Discovered by checking the checkpoint file directly (which is
# explicitly flushed) and finding real progress with zero visible stdout.
print = partial(print, flush=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = REPO_ROOT / "data" / "train.csv"
CHECKPOINT_FILE = REPO_ROOT / "data" / ".report_labels_checkpoint.jsonl"
SILVER_LABELS_OUT = REPO_ROOT / "data" / "report_labels_llm.csv"
CALIBRATION_OUT = REPO_ROOT / "data" / "report_labels_llm_calibration.json"

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:7b"

LABEL_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

SYSTEM_PROMPT = """You are assisting with weak-label extraction for a knee MRI research dataset. \
You will be given the free-text radiology report for one knee MRI study. Reports may be written \
in any language depending on the reporting institution (Spanish, Dutch, German, French, and others) \
— read and interpret the report regardless of language; do not translate anything in your answer.

Extract whether each of these 12 findings is present, as a binary 0/1 judgment:
- ACL: anterior cruciate ligament injury
- MCL: medial collateral ligament injury
- Medial Meniscus: medial meniscus tear
- Lateral Meniscus: lateral meniscus tear
- Medial OA: osteoarthritis of the medial tibiofemoral compartment
- Lateral OA: osteoarthritis of the lateral tibiofemoral compartment
- PF OA: patellofemoral osteoarthritis
- Effusion: joint effusion / excess fluid
- Synovitis: inflammation of the joint lining
- Baker's: Baker's cyst
- Contusion: bone contusion / bone bruise
- Fracture: fracture

Rules:
- Base your answer only on the report text provided — do not infer findings the report doesn't
  support, and do not use any outside knowledge about the patient.
- If a finding is not mentioned at all, treat it as negative (0) — this matches standard
  radiology reporting convention where an unmentioned structure is presumed normal.
- For hedged or uncertain language ("cannot exclude X", "possible X", "trace X"), default to
  negative (0) unless the report's own impression/conclusion section treats it as a real,
  actionable finding — borderline cases should be marked negative, matching how this dataset's
  directly-labeled studies were adjudicated by radiologists.
- The impression/conclusion section (when present) is more authoritative than the body of the
  report for borderline cases.
- Answer with exactly one 0 or 1 for each of the 12 fields — no other output."""


def _label_schema():
    return {
        "type": "object",
        "properties": {col: {"type": "integer", "enum": [0, 1]} for col in LABEL_COLUMNS},
        "required": LABEL_COLUMNS,
        "additionalProperties": False,
    }


def _load_train():
    if not TRAIN_CSV.exists():
        print(f"ERROR: {TRAIN_CSV} not found — run scripts/download_data.py first.",
              file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(TRAIN_CSV)
    gold_mask = df[LABEL_COLUMNS].notna().all(axis=1)
    return df, gold_mask


def _check_ollama(model):
    try:
        resp = requests.get("http://localhost:11434/api/version", timeout=5)
        resp.raise_for_status()
    except requests.exceptions.RequestException:
        print("ERROR: Ollama server not reachable at localhost:11434 — "
              "is the Ollama app running?", file=sys.stderr)
        sys.exit(1)

    tags = requests.get("http://localhost:11434/api/tags", timeout=5).json()
    have = {m["name"] for m in tags.get("models", [])}
    if model not in have:
        print(f"ERROR: model '{model}' not found locally. Pull it first: "
              f"`ollama pull {model}`. Locally available: {sorted(have)}", file=sys.stderr)
        sys.exit(1)


def _extract_one(model, report_text, timeout=180):
    """Returns (labels_dict_or_None, error_str_or_None)."""
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Radiology report:\n\n{report_text}"},
            ],
            "format": _label_schema(),
            "stream": False,
            "keep_alive": "60m",  # avoid reload cost across thousands of sequential calls
        }, timeout=timeout)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        data = json.loads(content)
        return {c: int(data[c]) for c in LABEL_COLUMNS}, None
    except requests.exceptions.RequestException as e:
        return None, f"request error: {e}"
    except (KeyError, json.JSONDecodeError, ValueError) as e:
        return None, f"parse error: {e}"


def _load_checkpoint():
    done = {}
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done[rec["StudyInstanceUID"]] = rec
    return done


def cmd_dry_run(args):
    df, gold_mask = _load_train()
    target = df.head(args.limit) if args.limit else df
    n_reports = len(target)
    # from the real timing test in this session: 17.6s for a 792-prompt-token
    # report on qwen2.5:7b, once the model is warm (keep_alive avoids reload cost)
    est_seconds_per_report = 17.6
    est_total = n_reports * est_seconds_per_report
    hrs = est_total / 3600
    print(f"Would process {n_reports} reports (of {len(df)} total, "
          f"{gold_mask.sum()} gold + {(~gold_mask).sum()} non-gold).")
    print(f"Estimated time at ~{est_seconds_per_report}s/report (measured, not guaranteed — "
          f"varies with report length and machine load): ~{est_total/60:.0f} min (~{hrs:.1f}h).")
    print("No cost — this runs entirely locally. Run `run` when ready.")


def cmd_run(args):
    _check_ollama(args.model)
    df, gold_mask = _load_train()
    target = df.head(args.limit) if args.limit else df

    done = _load_checkpoint()
    remaining = target[~target["StudyInstanceUID"].isin(done.keys())]
    print(f"{len(done)} already done (checkpoint), {len(remaining)} remaining "
          f"of {len(target)} targeted.")

    if remaining.empty:
        print("Nothing left to do — run `finalize` to write outputs.")
        return

    start = time.time()
    n_errors = 0
    with open(CHECKPOINT_FILE, "a") as f:
        for i, (_, row) in enumerate(remaining.iterrows()):
            labels, err = _extract_one(args.model, row["Report"])
            rec = {"StudyInstanceUID": row["StudyInstanceUID"], "labels": labels, "error": err}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if err:
                n_errors += 1
                print(f"  [{i+1}/{len(remaining)}] {row['StudyInstanceUID']}: ERROR {err}")
            elif (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - start
                rate = elapsed / (i + 1)
                eta_min = rate * (len(remaining) - i - 1) / 60
                print(f"  [{i+1}/{len(remaining)}] {rate:.1f}s/report avg, "
                      f"ETA ~{eta_min:.0f} min, {n_errors} errors so far")

    print(f"\nDone this run: {len(remaining)} processed, {n_errors} errors. "
          f"Run `finalize` to write outputs, or `run` again if interrupted early.")


def cmd_finalize(args):
    done = _load_checkpoint()
    if not done:
        print("No checkpoint data found — run `run` first.", file=sys.stderr)
        sys.exit(1)

    df, gold_mask = _load_train()
    gold_uids = set(df.loc[gold_mask, "StudyInstanceUID"])

    ok = {uid: rec["labels"] for uid, rec in done.items() if rec["labels"] is not None}
    errors = {uid: rec["error"] for uid, rec in done.items() if rec["labels"] is None}
    print(f"{len(ok)} successful extractions, {len(errors)} failed (excluded from output).")

    silver_rows = [{"StudyInstanceUID": uid, **labels}
                    for uid, labels in ok.items() if uid not in gold_uids]
    silver_df = pd.DataFrame(silver_rows, columns=["StudyInstanceUID"] + LABEL_COLUMNS)
    silver_df.to_csv(SILVER_LABELS_OUT, index=False)
    print(f"Wrote {len(silver_df)} silver-labeled studies to {SILVER_LABELS_OUT}")

    calib_rows = []
    for uid, labels in ok.items():
        if uid in gold_uids:
            true_row = df.loc[df["StudyInstanceUID"] == uid, LABEL_COLUMNS].iloc[0]
            calib_rows.append({"StudyInstanceUID": uid,
                                **{f"true_{c}": int(true_row[c]) for c in LABEL_COLUMNS},
                                **{f"pred_{c}": labels[c] for c in LABEL_COLUMNS}})
    if calib_rows:
        calib_df = pd.DataFrame(calib_rows)
        from sklearn.metrics import precision_recall_fscore_support
        per_label = {}
        for c in LABEL_COLUMNS:
            y_true = calib_df[f"true_{c}"].values
            y_pred = calib_df[f"pred_{c}"].values
            p, r, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0)
            per_label[c] = {"precision": p, "recall": r, "f1": f1,
                             "agreement": float((y_true == y_pred).mean()),
                             "n_positive_true": int(y_true.sum())}
        overall_agreement = float(
            (calib_df[[f"true_{c}" for c in LABEL_COLUMNS]].values
             == calib_df[[f"pred_{c}" for c in LABEL_COLUMNS]].values).mean()
        )
        report = {"n_calibration_studies": len(calib_df), "overall_agreement": overall_agreement,
                  "model": args.model if hasattr(args, "model") else DEFAULT_MODEL,
                  "per_label": per_label}
        with open(CALIBRATION_OUT, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nCalibration against {len(calib_df)} gold-labeled studies "
              f"(wrote {CALIBRATION_OUT}):")
        print(f"  overall cell agreement: {overall_agreement:.3f}")
        for c, m in per_label.items():
            print(f"  {c:20s} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} "
                  f"(n_pos={m['n_positive_true']})")
    else:
        print("No gold studies were in this run — no calibration report produced.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_dry = sub.add_parser("dry-run", help="Estimate time, no cost, nothing submitted")
    p_dry.add_argument("--limit", type=int, default=None)
    p_dry.set_defaults(func=cmd_dry_run)

    p_run = sub.add_parser("run", help="Process reports sequentially (resumable)")
    p_run.add_argument("--limit", type=int, default=None,
                        help="Only process the first N studies (from the full train.csv)")
    p_run.add_argument("--model", default=DEFAULT_MODEL)
    p_run.set_defaults(func=cmd_run)

    p_fin = sub.add_parser("finalize", help="Write silver labels + calibration from checkpoint")
    p_fin.add_argument("--model", default=DEFAULT_MODEL)
    p_fin.set_defaults(func=cmd_finalize)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
