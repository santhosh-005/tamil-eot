#!/usr/bin/env python3
"""Step 1f -- draw a blind listening sample so the label accuracy is measured
rather than assumed.

Every number this project reports rests on one claim: that a channel-derived
label matches what a Tamil speaker hears. That claim is cheap to state and
nobody should believe it without evidence, so this exports a stratified sample
in a blind protocol.

  reports/qa/blind/<n>.wav   the clip exactly as a model would see it
  reports/qa/sheet.tsv       one row per clip, verdict column empty
  reports/qa/answers.tsv     the labels, kept separate

Listen to a clip, write complete / incomplete / unsure in the verdict column,
and only then open answers.tsv. Scoring sheet against answers gives the label
accuracy of the whole pipeline.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tamileot import audio, rules  # noqa: E402
from tamileot.paths import CLIPS, GOLD, REPORTS  # noqa: E402

SEED = 20260819


def bucket(r: dict) -> tuple[int, str]:
    g = r["gap"]
    if g < 0.35:
        b = "fast"
    elif g < 0.8:
        b = "mid"
    else:
        b = "slow"
    return r["label"], b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--split", default="train", help="draw from this split, so test stays untouched")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (GOLD / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if rules.is_core(r) and r["split"] == args.split]

    strata: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for r in rows:
        strata[bucket(r)].append(r)

    rng = random.Random(SEED)
    per = max(1, args.n // len(strata))
    pick: list[dict] = []
    for k in sorted(strata):
        pool = strata[k]
        rng.shuffle(pool)
        pick.extend(pool[:per])
    # spread the draw across calls rather than concentrating on the chatty ones
    rng.shuffle(pick)
    pick = pick[: args.n]

    out = REPORTS / "qa"
    blind = out / "blind"
    blind.mkdir(parents=True, exist_ok=True)
    for f in blind.glob("*.wav"):
        f.unlink()

    sheet = ["idx\tverdict\tnotes"]
    ans = ["idx\tlabel\tsource\tgap_s\tcall\tside\tt\ttext"]
    for i, r in enumerate(pick, 1):
        src = CLIPS / r["split"] / f"{r['sid']}.wav"
        audio.write(blind / f"{i:03d}.wav", audio.read(src))
        sheet.append(f"{i}\t\t")
        lab = "complete" if r["label"] == 1 else "incomplete"
        ans.append(f"{i}\t{lab}\t{r['source']}\t{r['gap']}\t{r['base']}\t{r['side']}\t{r['t']}\t{r['text']}")

    (out / "sheet.tsv").write_text("\n".join(sheet) + "\n", encoding="utf-8")
    (out / "answers.tsv").write_text("\n".join(ans) + "\n", encoding="utf-8")

    n_pos = sum(1 for r in pick if r["label"] == 1)
    print(f"drew {len(pick)} clips from '{args.split}' "
          f"({n_pos} complete, {len(pick) - n_pos} incomplete) "
          f"across {len({r['base'] for r in pick})} calls")
    for k in sorted(strata):
        got = sum(1 for r in pick if bucket(r) == k)
        print(f"  label={k[0]} gap={k[1]:5s}: {got:3d} drawn of {len(strata[k]):,} available")
    print(f"\n  {blind}/001.wav ...")
    print(f"  {out / 'sheet.tsv'}   <- fill the verdict column")
    print(f"  {out / 'answers.tsv'} <- do not open first")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
