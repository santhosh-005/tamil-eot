#!/usr/bin/env python3
"""Step 1g-ter -- rescore the bake-off offline, after a human label changed.

The 197 blind-listened clips are the only ground truth in this project, so a
wrong one moves every agreement number downstream. Step 11 flagged three clips
where all seven models contradicted the human in the same direction; they were
re-listened on 2026-08-24 and the models were right. `reports/qa/sheet.tsv`
now carries the corrected verdicts, and this recomputes everything keyed to it.

`11_label.py --collect` cannot: the GCP project holding the job output was deleted, so
the token and cost columns are unrecoverable and are carried forward below as
frozen literals from the 2026-08-20 run. Everything the correction actually
touches -- agreement, CI, per-class, precision, McNemar, consensus -- comes
from the local verdict caches, which survived.

    python pipeline/12_bakeoff.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tamileot.labelling import PROMPT_ID, human_labels, load_rows, qa_index, read_lines  # noqa: E402
from tamileot.paths import GOLD, REPORTS  # noqa: E402

# The scoring and report-writing live in step 11, which starts with a digit and
# so cannot be imported by name. Loaded by path rather than duplicated: this
# script exists to reproduce step 11's report from the local caches, and two
# copies of the scoring would eventually disagree about what the numbers mean.
_spec = importlib.util.spec_from_file_location("step11", ROOT / "pipeline/11_label.py")
_label = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_label)

# Measured 2026-08-20, on the jobs whose output no longer exists. Cost is
# unaffected by a label correction, so it is carried rather than recomputed --
# and marked as carried in the report, because a number nothing can re-derive
# should say so.
BILLED = {
    #                          $ run   tok_in  tok_out  think  $9,406  $16,216  err
    "gemini-3.7-flash":       (0.07,   944,    147,     120,   3.35,   5.77,    0),
    "gemini-3.1-pro-preview": (0.50,   944,    375,     338,   23.88,  41.17,   0),
    "gemini-3.6-flash":       (0.11,   944,    312,     282,   5.28,   9.10,    0),
    "gemini-3.5-flash":       (0.14,   944,    414,     377,   6.86,   11.82,   0),
    "gemini-3-flash-preview": (0.21,   944,    512,     451,   9.91,   17.09,   0),
    "gemini-2.5-flash":       (0.16,   944,    497,     464,   7.83,   13.50,   1),
    "gemini-3.5-flash-lite":  (0.01,   944,    29,      0,     0.69,   1.18,    0),
}


def load_cache(model: str, keep: set[str]) -> dict[str, dict]:
    got = {}
    for line in read_lines(f"batch_{model}_{PROMPT_ID}.jsonl"):
        r = json.loads(line)
        if r["sid"] in keep:
            got[r["sid"]] = r
    return got


def main() -> int:
    hum = human_labels()
    rows = {r["sid"]: r for r in load_rows("samples.jsonl")}
    idx = qa_index()

    table, per_model = [], {}
    for model in BILLED:
        got = load_cache(model, set(hum))
        st = _label.stats(got, hum, rows)
        st["model"] = model
        st["labelled"] = len(got)
        table.append(st)
        per_model[model] = got
        _label.write_sheet(model, got, hum, rows, idx)
    table.sort(key=lambda d: -d["acc"])
    per_model = {d["model"]: per_model[d["model"]] for d in table}

    binar = {mo: {s: 1 if g["verdict"] == "complete" else 0
                  for s, g in gt.items() if g["verdict"] in ("complete", "incomplete")}
             for mo, gt in per_model.items()}
    csv_out = _label.write_combined(per_model, hum, rows, idx)

    pipe = _label.pipeline_stats(hum, rows)
    top = table[0]
    n = top["n"]

    L = [f"# Labeller bake-off — {len(hum)} human-labelled clips, Vertex AI Batch", "",
         f"Prompt `{PROMPT_ID}`. Audio only, no transcript. Same clips, same human",
         "labels, every model — so the comparison is paired.", "",
         f"**Winner: `{top['model']}` — {100 * top['acc']:.1f}% agreement, "
         f"{100 * top['prec_inc']:.1f}% precision on `incomplete`, "
         f"${BILLED[top['model']][5]:.2f} for the whole pool.**", "",
         f"Bar was 90%. {sum(d['acc'] >= 0.90 for d in table)} of {len(table)} clear it. "
         "Ties at the top broken on price.", "",
         "| model | n | agreement | 95% CI | `change` | `hold_intra` | prec `incomplete` | $ run | in/out tok | think |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for d in table:
        b = BILLED[d["model"]]
        L.append(f"| `{d['model']}` | {d['n']} | **{100 * d['acc']:.1f}%** | "
                 f"{100 * d['acc_ci'][0]:.0f}–{100 * d['acc_ci'][1]:.0f}% | "
                 f"{100 * d['change']:.1f}% | {100 * d['hold_intra']:.1f}% | "
                 f"{100 * d['prec_inc']:.1f}% ({d['prec_inc_n']}) | "
                 f"${b[0]:.2f} | {b[1]}/{b[2]} | {b[3]} |")
    L += [f"| *our pipeline* | {len(hum)} | {100 * pipe['all']:.1f}% | – | "
          f"{100 * pipe['change']:.1f}% | {100 * pipe['hold_intra']:.1f}% | "
          f"{100 * pipe['prec_inc']:.1f}% | – | – | – |", "",
          "Cost columns are carried from the 2026-08-20 run — the GCP project holding",
          "the job output is gone, and a label correction does not change what was billed.",
          "Agreement columns are recomputed from the local verdict caches.", ""]

    best = top["model"]
    L += [f"## Is the gap real? Paired McNemar against `{best}`", "",
          "Discordant pairs only — clips where exactly one of the two was right.", "",
          "| vs | winner-only right | other-only right | p |", "|---|---|---|---|"]
    for d in table[1:]:
        b01, b10, p = _label.mcnemar(binar[best], binar[d["model"]], hum)
        sig = "**significant**" if p < 0.05 else "not significant"
        L.append(f"| `{d['model']}` | {b01} | {b10} | {p:.3f} — {sig} |")

    consensus = []
    for sid, h in hum.items():
        votes = [b[sid] for b in binar.values() if sid in b]
        if len(votes) >= 5 and sum(v != h for v in votes) >= len(votes) - 1:
            consensus.append((sid, h, sum(v != h for v in votes), len(votes)))
    L += ["", "## Clips where the models agree against the human label", ""]
    if consensus:
        wrong_here = sum(1 for sid, *_ in consensus if binar[best].get(sid) != hum[sid])
        L += [f"{len(consensus)} of {len(hum)} clips have all-but-at-most-one model contradicting",
              f"the human, same direction — {wrong_here} of `{best}`'s "
              f"{n - round(top['acc'] * n)} errors. Re-listen candidates.", "",
              "| clip | human said | models against | of |", "|---|---|---|---|"]
        for sid, h, k, tot in sorted(consensus, key=lambda x: -x[2]):
            L.append(f"| `{idx.get(sid, 0):03d}.wav` | "
                     f"{'complete' if h == 1 else 'incomplete'} | {k} | {tot} |")
    else:
        L += ["None. The reference is one human's single listen to an isolated 8 s",
              "clip with no future audio, so a clip every model calls the other way is",
              "evidence against the label. Three were, and `reports/qa/sheet.tsv` now",
              "holds the corrected verdicts."]

    L += ["", "## Cost to label the full pool at batch rates", "",
          "| model | `hold_intra` 9,406 | everything 16,216 |", "|---|---|---|"]
    for d in table:
        b = BILLED[d["model"]]
        L.append(f"| `{d['model']}` | ${b[4]:.2f} | ${b[5]:.2f} |")
    L += ["", "Errors / unparsed per model: "
          + ", ".join(f"`{d['model']}` {BILLED[d['model']][6]}" for d in table), "",
          "Regenerate with `python pipeline/12_bakeoff.py`.", ""]

    out = REPORTS / "model_bakeoff.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n  -> {out}\n  -> {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
