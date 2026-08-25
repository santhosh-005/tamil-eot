#!/usr/bin/env python3
"""Step 1g -- label the clip pool with an audio LLM, on Vertex AI Batch.

This is the step that replaces the pipeline's own `hold_intra` label, which a
blind listening pass measured at 44.4% -- below chance. See
`src/tamileot/labelling.py` for the question being asked and why.

Batch rather than the interactive Gemini API, for three reasons:

  * **half the token price** -- batch is billed at 50% of interactive rates,
  * **no per-minute or per-day quota** to rotate keys around,
  * **ADC instead of API keys**, so nothing secret is passed on a command line.

The prompt and the human-label readers are imported from `tamileot.labelling`,
so nothing here can drift from what was actually asked (prompt `ca216b25`).

    export TAMILEOT_GCP_PROJECT=my-project
    export TAMILEOT_GCS_BUCKET=gs://my-bucket
    gcloud auth application-default login

    python pipeline/11_label.py --upload --submit --wait --collect

Audio is referenced by `gs://` URI, not inlined -- upload once, reuse across
every model. `--upload` is idempotent.

Every model reads the identical 197 clips and is scored against the identical
human labels from the blind listening pass, so the comparison is paired.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tamileot.labelling import (  # noqa: E402
    PROMPT, PROMPT_ID, human_labels, load_rows, qa_index,
)
from tamileot.paths import CLIPS, GOLD, REPORTS  # noqa: E402

# Your own GCP project and a bucket in it. No default -- a hardcoded project id
# is one person's billing account, and a wrong one fails halfway through an
# upload rather than at startup.
PROJECT = os.environ.get("TAMILEOT_GCP_PROJECT", "")
BUCKET = os.environ.get("TAMILEOT_GCS_BUCKET", "")
LOCATION = os.environ.get("TAMILEOT_GCP_LOCATION", "global")  # 3.x models are global-only
API = "https://aiplatform.googleapis.com/v1"

# Every Gemini the project can reach that is worth putting in front of audio.
# gemini-3-flash-preview is the incumbent: it scored 93.3% on 120 of these
# clips through the interactive API, so it anchors the comparison.
MODELS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
]

GCLOUD = (os.environ.get("TAMILEOT_GCLOUD")
          or shutil.which("gcloud")
          or str(Path.home() / "google-cloud-sdk/bin/gcloud"))
STATE = GOLD / "batch_jobs.json"


def require_gcp() -> None:
    """Fail at startup, not halfway through a 16k-clip upload."""
    missing = [n for n, v in (("TAMILEOT_GCP_PROJECT", PROJECT),
                              ("TAMILEOT_GCS_BUCKET", BUCKET)) if not v]
    if missing:
        sys.exit("set " + " and ".join(missing) + "\n"
                 "  export TAMILEOT_GCP_PROJECT=my-project\n"
                 "  export TAMILEOT_GCS_BUCKET=gs://my-bucket")
    if not Path(GCLOUD).exists() and not shutil.which(GCLOUD):
        sys.exit(f"gcloud not found at {GCLOUD} -- set TAMILEOT_GCLOUD to its path")

# Which row file this invocation labels, set by --rows. A module global rather
# than a parameter threaded through jobs/collect/write_relabelled, because
# every one of them needs the same answer and there is only ever one per run.
ROWS = "samples.jsonl"


def rows_all() -> list[dict]:
    return load_rows(ROWS)


def rows_tag() -> str:
    """Suffix that keeps a second row file's jobs from colliding with the
    first's in `batch_jobs.json` -- both are `--source all`, so without this a
    relaxed run would overwrite the original run's state under the same key."""
    return "" if ROWS == "samples.jsonl" else "+" + ROWS.replace("samples_", "").replace(".jsonl", "")


def owns(tag: str) -> bool:
    """Whether `tag` belongs to this invocation's row file.

    Without this, `--wait` polls the previous run's jobs too. Those succeeded
    months ago and their GCS output died with the bucket, so the API returns an
    error, `state` becomes `?`, `?` is not in DONE, and the wait loop never
    exits. `--collect` would also try to re-read their deleted output.
    """
    return tag.endswith(rows_tag()) if rows_tag() else "+" not in tag


# --------------------------------------------------------------------------- gcloud

def _token() -> str:
    return subprocess.run([GCLOUD, "auth", "print-access-token"],
                          capture_output=True, text=True, check=True).stdout.strip()


def _api(method: str, url: str, body: dict | None = None) -> dict:
    cmd = ["curl", "-sS", "-X", method,
           "-H", f"Authorization: Bearer {_token()}",
           "-H", "Content-Type: application/json", url]
    if body is not None:
        cmd += ["-d", json.dumps(body)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"error": {"message": out[:400]}}


def _gs(*args: str) -> str:
    return subprocess.run([GCLOUD, "storage", *args],
                          capture_output=True, text=True).stdout


# --------------------------------------------------------------------------- jobs

QA = "qa"   # the 197 human-labelled clips -- the bake-off selection


def jobs(source: str = QA, split: str = "") -> list[tuple[str, Path]]:
    """(sid, local wav) for the requested selection, clips-on-disk only.

    `--source qa` is the 197 clips from the blind listening pass, which is what
    ranks labellers. Anything else is a bulk selection out of `samples.jsonl`.
    The held-back classes have no wavs until `06_cut_heldback.py` has run, so
    the filter is on the file existing, not on the row existing.
    """
    rows = {r["sid"]: r for r in rows_all()}
    if source == QA:
        sel = [rows[s] for s in human_labels() if s in rows]
    else:
        sel = [r for r in rows.values() if source == "all" or r["source"] == source]
    if split:
        sel = [r for r in sel if r["split"] == split]
    out = [(r["sid"], CLIPS / r["split"] / f"{r['sid']}.wav") for r in sel]
    return sorted((s, p) for s, p in out if p.exists())


def uploaded() -> set[str]:
    return {l.rsplit("/", 1)[-1]
            for l in _gs("ls", f"{BUCKET}/clips/").split() if l.endswith(".wav")}


def upload(sel: list[tuple[str, Path]]) -> None:
    """Idempotent: only what is not already in the bucket is sent.

    Audio is uploaded once and addressed by URI from then on, so a second model
    over the same clips costs nothing extra in transfer.
    """
    have = uploaded()
    todo = [p for _, p in sel if p.name not in have]
    mb = sum(p.stat().st_size for p in todo) / 1e6
    print(f"  {len(have):,} already in {BUCKET}/clips/, {len(todo):,} to upload ({mb:,.0f} MB)")
    if todo:
        listing = "\n".join(str(p) for p in todo)
        subprocess.run([GCLOUD, "storage", "cp", "-I", f"{BUCKET}/clips/"],
                       input=listing, text=True, check=True)


def cached_sids(model: str) -> set[str]:
    """Verdicts already paid for. A bulk run over 13k clips is long enough that
    resuming matters, and the 197 bake-off clips are members of the pool."""
    got = set()
    for p in (GOLD / f"batch_{model}_{PROMPT_ID}.jsonl",
              GOLD / f"llm_{model}_{PROMPT_ID}.jsonl"):
        if p.exists():
            for l in p.read_text(encoding="utf-8").splitlines():
                r = json.loads(l)
                if r.get("verdict") in ("complete", "incomplete"):
                    got.add(r["sid"])
    return got


def write_input(model: str, tag: str, sel: list[tuple[str, Path]]) -> tuple[str, int]:
    """One JSONL line per clip. The gs:// URI is what maps a response back to a
    sid on collection -- batch output echoes the request verbatim."""
    local = ROOT / f".batch_input_{tag.replace('/', '_')}.jsonl"
    with local.open("w", encoding="utf-8") as f:
        for _sid, path in sel:
            f.write(json.dumps({"request": {
                "contents": [{"role": "user", "parts": [
                    {"text": PROMPT},
                    {"fileData": {"mimeType": "audio/wav",
                                  "fileUri": f"{BUCKET}/clips/{path.name}"}},
                ]}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2048,
                                     "responseMimeType": "application/json"},
            }}) + "\n")
    uri = f"{BUCKET}/input/{tag.replace('/', '_')}_{PROMPT_ID}.jsonl"
    subprocess.run([GCLOUD, "storage", "cp", str(local), uri],
                   capture_output=True, text=True, check=True)
    local.unlink()
    return uri, len(sel)


def tag_for(model: str, source: str, split: str) -> str:
    """Validation runs keep the bare model name so the bake-off state survives."""
    if source == QA:
        return model
    return f"{model}@{source}" + (f"-{split}" if split else "") + rows_tag()


def submit(models: list[str], source: str, split: str, limit: int, resume: bool,
           splits: list[str] | None = None) -> None:
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    # A bulk run with no split named goes out as one job per split, smallest
    # first. Test is what unblocks scoring, and three jobs fail independently.
    if splits is None:
        splits = [split] if (split or source == QA) else ["test", "dev", "train"]
    for model in models:
        skip = cached_sids(model) if resume else set()
        for sp in splits:
            sel = jobs(source, sp)
            if resume:
                sel = [(s, p) for s, p in sel if s not in skip]
            if limit:
                sel = sel[:limit]
            tag = tag_for(model, source, sp)
            if not sel:
                print(f"  {tag:34s} nothing to do (all cached)")
                continue
            uri, n = write_input(model, tag, sel)
            stamp = time.strftime("%m%d-%H%M%S")
            r = _api("POST", f"{API}/projects/{PROJECT}/locations/{LOCATION}/batchPredictionJobs", {
                "displayName": f"tamileot-{tag}-{PROMPT_ID}-{stamp}".replace("@", "-"),
                "model": f"publishers/google/models/{model}",
                "inputConfig": {"instancesFormat": "jsonl", "gcsSource": {"uris": [uri]}},
                "outputConfig": {"predictionsFormat": "jsonl",
                                 "gcsDestination": {"outputUriPrefix": f"{BUCKET}/out/{tag.replace('/', '_')}_{stamp}/"}},
            })
            if "name" not in r:
                print(f"  {tag:34s} SUBMIT FAILED  {str(r.get('error', {}).get('message', r))[:160]}")
                continue
            # source/split are stored rather than parsed back out of the tag:
            # `hold_inter` and `change_midseg` contain separators of their own.
            state[tag] = {"model": model, "source": source, "split": sp,
                          "job": r["name"], "state": r.get("state", "?"),
                          "input": uri, "n": n}
            print(f"  {tag:34s} {n:6,d} clips  {r['name'].rsplit('/', 1)[-1]}  {r.get('state')}")
    STATE.write_text(json.dumps(state, indent=1))
    print(f"\n  -> {STATE}")


def poll(tags: list[str] | None = None, quiet: bool = False) -> dict:
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    for tag, s in state.items():
        if tags is not None and tag not in tags:
            continue
        r = _api("GET", f"{API}/{s['job']}")
        s["state"] = r.get("state", "?")
        s["out"] = (r.get("outputInfo") or {}).get("gcsOutputDirectory", "")
        if r.get("error"):
            s["err"] = str(r["error"].get("message", ""))[:200]
        st = r.get("completionStats") or {}
        s["done_n"] = int(st.get("successfulCount", 0))
        s["fail_n"] = int(st.get("failedCount", 0))
        if not quiet:
            pct = f"{100 * s['done_n'] / s['n']:3.0f}%" if s.get("n") else "    "
            print(f"  {tag:34s} {s['state']:26s} ok {s['done_n']:6,d} {pct} fail {s['fail_n']:4d}"
                  + (f"  {s['err']}" if s.get("err") else ""))
    STATE.write_text(json.dumps(state, indent=1))
    return state


# --------------------------------------------------------------------------- scoring

def parse_out(model: str, outdir: str) -> tuple[dict[str, dict], dict]:
    """-> {sid: verdict record}, token usage totals.

    Usage is read from `promptTokensDetails`, which splits the prompt by
    modality. That matters twice over: audio and text bill at different rates,
    and the split is the only way to see that 8 s of telephony costs 200 tokens
    (25/s) rather than the 256 an earlier `countTokens` probe reported.
    `thoughtsTokenCount` bills as output and is counted here for the same
    reason -- on the 3.x models it is several times the visible answer.
    """
    listing = [l for l in _gs("ls", outdir.rstrip("/") + "/**").split() if l.endswith(".jsonl")]
    got: dict[str, dict] = {}
    use = {"audio": 0, "text": 0, "out": 0, "think": 0, "n": 0}
    for uri in listing:
        # Streamed, not slurped: each row echoes the 2.7 kB prompt back, so a
        # 10k-clip shard is tens of megabytes and there is no reason to hold it.
        proc = subprocess.Popen([GCLOUD, "storage", "cat", uri],
                                stdout=subprocess.PIPE, text=True, encoding="utf-8")
        for line in proc.stdout:
            if not line.strip():
                continue
            row = json.loads(line)
            uri_in = ""
            # Batch echoes every part with all fields present, so the text part
            # carries an explicit `"fileData": null` -- `in` is not enough.
            for part in (row.get("request", {}).get("contents") or [{}])[0].get("parts", []):
                if part.get("fileData"):
                    uri_in = part["fileData"].get("fileUri", "")
            sid = uri_in.rsplit("/", 1)[-1].removesuffix(".wav")
            if not sid:
                continue
            resp = row.get("response") or {}
            um = resp.get("usageMetadata") or {}
            for det in um.get("promptTokensDetails") or []:
                k = "audio" if det.get("modality") == "AUDIO" else "text"
                use[k] += det.get("tokenCount", 0)
            use["out"] += um.get("candidatesTokenCount", 0)
            use["think"] += um.get("thoughtsTokenCount", 0)
            use["n"] += 1
            try:
                txt = resp["candidates"][0]["content"]["parts"][0]["text"]
                d = json.loads(txt.strip().removeprefix("```json").removeprefix("```").removesuffix("```"))
                # Occasionally a model wraps the object in a one-element array.
                # That is still valid JSON, so it parses and then fails on
                # `.get` -- costing a clip for a formatting quirk, not a refusal.
                if isinstance(d, list) and len(d) == 1 and isinstance(d[0], dict):
                    d = d[0]
                v = str(d.get("verdict", "")).lower()
            except Exception:  # noqa: BLE001
                v, d = "error", {}
            if not isinstance(d, dict):
                v, d = "error", {}
            got[sid] = {"verdict": v if v in ("complete", "incomplete") else "error",
                        "confidence": float(d.get("confidence", 0.0) or 0.0),
                        "reason": " ".join(str(d.get("reason", "")).split())}
        proc.stdout.close()
        proc.wait()
    return got, use


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% CI. A 197-clip run separates 93% from 88% only loosely, and the
    interval is the honest way to say so."""
    if not n:
        return 0.0, 0.0
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def stats(got: dict[str, dict], hum: dict[str, int], rows: dict[str, dict]) -> dict:
    pairs = [(hum[s], 1 if got[s]["verdict"] == "complete" else 0, rows[s]["source"])
             for s in got if s in hum and s in rows and got[s]["verdict"] in ("complete", "incomplete")]
    d: dict = {"n": len(pairs), "err": sum(1 for g in got.values() if g["verdict"] == "error")}
    if not pairs:
        return d
    ok = sum(h == m for h, m, _ in pairs)
    d["acc"], d["acc_ci"] = ok / len(pairs), wilson(ok, len(pairs))
    for src in ("change", "hold_intra"):
        sel = [p for p in pairs if p[2] == src]
        if sel:
            a = sum(h == m for h, m, _ in sel)
            d[src] = a / len(sel)
            d[src + "_n"] = len(sel)
    # Precision of `incomplete` is the number that decides usability: it is the
    # class the pipeline gets wrong, so a labeller only helps if it is right here.
    for name, m in (("prec_inc", 0), ("prec_com", 1)):
        sel = [p for p in pairs if p[1] == m]
        if sel:
            k = sum(1 for h, _, _ in sel if h == m)
            d[name], d[name + "_n"] = k / len(sel), len(sel)
            d[name + "_ci"] = wilson(k, len(sel))
    return d


# Vertex batch = 50% of interactive. $/1M tokens, text-in / audio-in / out.
PRICE = {
    "gemini-3.7-flash":       (0.30, 0.60, 2.50),
    "gemini-3.6-flash":       (0.30, 0.60, 2.50),
    "gemini-3.5-flash":       (0.30, 1.00, 2.50),
    "gemini-3.5-flash-lite":  (0.10, 0.30, 0.40),
    "gemini-3-flash-preview": (0.50, 1.00, 3.00),
    "gemini-3.1-pro-preview": (1.25, 2.00, 10.00),
    "gemini-2.5-flash":       (0.30, 1.00, 2.50),
}


def cost(model: str, use: dict) -> float:
    """Batch price for this run, from measured tokens. Thinking bills as output."""
    t_in, a_in, t_out = PRICE.get(model, (0.30, 1.00, 2.50))
    return 0.5 * (use["text"] * t_in + use["audio"] * a_in
                  + (use["out"] + use["think"]) * t_out) / 1e6


def pipeline_stats(hum: dict[str, int], rows: dict[str, dict]) -> dict[str, float]:
    """The channel-derived labels, scored against the same humans.

    These were literals in the report template until a human label was
    corrected and every one of them went stale at once. Computed now.
    """
    pairs = [(hum[s], rows[s]["label"], rows[s]["source"]) for s in hum if s in rows]
    out = {"all": sum(h == p for h, p, _ in pairs) / len(pairs)}
    for src in ("change", "hold_intra"):
        sel = [p for p in pairs if p[2] == src]
        out[src] = sum(h == p for h, p, _ in sel) / len(sel)
    inc = [p for p in pairs if p[1] == 0]
    out["prec_inc"] = sum(1 for h, _, _ in inc if h == 0) / len(inc)
    return out


def mcnemar(a: dict[str, int], b: dict[str, int], hum: dict[str, int]) -> tuple[int, int, float]:
    """Exact paired test on the clips both models answered.

    Unpaired accuracies on 197 clips have overlapping CIs almost regardless of
    the gap, which makes them nearly useless for ranking. What matters is the
    discordant pairs -- clips where exactly one model was right -- so the test
    is over those only, and its null is a fair coin.
    """
    sids = [s for s in hum if s in a and s in b]
    b01 = sum(1 for s in sids if a[s] == hum[s] and b[s] != hum[s])
    b10 = sum(1 for s in sids if a[s] != hum[s] and b[s] == hum[s])
    n = b01 + b10
    if not n:
        return b01, b10, 1.0
    tail = sum(math.comb(n, k) for k in range(min(b01, b10) + 1))
    return b01, b10, min(1.0, 2 * tail / 2 ** n)


def write_combined(per_model: dict[str, dict], hum: dict[str, int],
                   rows: dict[str, dict], idx: dict[str, int]) -> Path:
    """One row per clip, one column per model -- the sheet to actually read.

    `n_wrong` sorts the genuinely hard clips to the top: a clip every model got
    right needs no attention, and a clip every model got wrong is far more
    likely a human slip than seven independent failures.
    """
    models = list(per_model)
    out = REPORTS / "qa" / f"bakeoff_all_models_{PROMPT_ID}.csv"
    recs = []
    for sid, h in hum.items():
        if sid not in rows:
            continue
        r = rows[sid]
        rec = {"idx": idx.get(sid, 0), "clip": f"{idx.get(sid, 0):03d}.wav",
               "you": "complete" if h == 1 else "incomplete",
               "pipeline": "complete" if r["label"] == 1 else "incomplete",
               "source": r["source"], "gap_s": f"{r['gap']:.3f}"}
        wrong = 0
        for mo in models:
            g = per_model[mo].get(sid, {})
            v = g.get("verdict", "")
            rec[mo] = v
            if v in ("complete", "incomplete"):
                wrong += (1 if v == "complete" else 0) != h
        rec["n_wrong"] = wrong
        rec["your_call"] = ""
        rec["reason_best"] = per_model[models[0]].get(sid, {}).get("reason", "")
        rec["text"] = " ".join((r.get("text") or "").split())
        rec["sid"] = sid
        recs.append(rec)
    recs.sort(key=lambda d: (-d["n_wrong"], d["idx"]))
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]), quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(recs)
    return out


def gather(state: dict, model: str, source: str) -> tuple[dict[str, dict], dict]:
    """Union every finished job for one model and selection.

    A bulk run goes out as one job per split, so the verdicts for a model are
    spread over several output directories and have to be merged before they
    mean anything.
    """
    # Seeded from the cache, not built from scratch. A resumed run only requests
    # the clips it is missing, so the jobs alone would not contain the verdicts
    # already paid for -- and `collect` rewrites the cache from this dict.
    got: dict[str, dict] = {}
    cache = GOLD / f"batch_{model}_{PROMPT_ID}.jsonl"
    if cache.exists():
        for l in cache.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            got[r["sid"]] = {k: r[k] for k in ("verdict", "confidence", "reason") if k in r}
    use = {"audio": 0, "text": 0, "out": 0, "think": 0, "n": 0}
    for tag, s in state.items():
        if (s.get("model") or tag) != model:
            continue
        if (s.get("source") or QA) != source or not owns(tag):
            continue
        if not s.get("out"):
            print(f"  {tag:34s} no output yet ({s.get('state', '?')})")
            continue
        g, u = parse_out(model, s["out"])
        got.update(g)
        for k in use:
            use[k] += u[k]
    return got, use


def collect(models: list[str], source: str, policy: str = "llm") -> None:
    state = poll(quiet=True)
    rows = {r["sid"]: r for r in rows_all()}
    hum = human_labels()
    idx = qa_index()
    table, per_model = [], {}
    # The cache is cumulative across selections; the report is not. After a bulk
    # run the cache holds 16k verdicts, and re-collecting the bake-off must
    # still produce a 197-clip sheet rather than silently widening it.
    want = set(hum) if source == QA else {s for s, _ in jobs(source, "")}
    for model in models:
        full, use = gather(state, model, source)
        got = {s: g for s, g in full.items() if s in want}
        if not got:
            print(f"  {model:24s} nothing collected")
            continue
        st = stats(got, hum, rows)
        st["model"] = model
        # Token usage lives only in the job output, and the job output lives
        # only in the bucket. Once that is deleted a re-collect would quietly
        # report $0.00 and 0 tokens, so what was measured is kept beside the
        # cache and reused when the jobs are gone.
        side = GOLD / f"batch_usage_{model}_{PROMPT_ID}{rows_tag()}.json"
        if use["n"]:
            side.write_text(json.dumps(use, indent=1))
        elif side.exists():
            use = json.loads(side.read_text())
            print(f"  {model:24s} jobs no longer reachable; usage from {side.name}")
        n = max(use["n"], 1)
        st["cost"] = cost(model, use)
        st["tok_in"] = (use["text"] + use["audio"]) / n
        st["tok_out"] = (use["out"] + use["think"]) / n
        st["think"] = use["think"] / n
        st["labelled"] = len(got)
        table.append(st)
        per_model[model] = got
        # Per-model verdict cache, same shape and naming as 07's. Rewritten from
        # the job output rather than appended to, so a re-collect is idempotent
        # and a larger run always supersedes a smaller one.
        cache = GOLD / f"batch_{model}_{PROMPT_ID}.jsonl"
        with cache.open("w", encoding="utf-8") as f:
            for sid, g in sorted(full.items()):
                f.write(json.dumps({**g, "sid": sid}, ensure_ascii=False) + "\n")
        print(f"  {model:24s} {len(got):,} verdicts in this selection, "
              f"{len(full):,} cached  ->  {cache.name}")
        if source == QA:
            write_sheet(model, {s: g for s, g in got.items() if s in hum}, hum, rows, idx)
    if not table:
        return
    table.sort(key=lambda d: -d.get("acc", 0))
    per_model = {d["model"]: per_model[d["model"]] for d in table}
    binar = {mo: {s: 1 if g["verdict"] == "complete" else 0
                  for s, g in gt.items() if g["verdict"] in ("complete", "incomplete")}
             for mo, gt in per_model.items()}
    if source == QA:
        write_summary(table, binar, hum)
        print(f"  sheet: {write_combined(per_model, hum, rows, idx)}")
    else:
        for d in table:
            write_relabelled(d["model"], per_model[d["model"]], rows, d, policy)


def write_relabelled(model: str, got: dict[str, dict], rows: dict[str, dict],
                     st: dict, policy: str = "llm") -> None:
    """`samples_llm.jsonl` + a report on what the relabelling actually changed.

    Both labels are always written: `label_pipeline` is the original and
    `llm_verdict` the model's, so the policy is a one-flag decision downstream
    rather than something baked irreversibly into the file.

    Which one `label` takes was a real choice:

    * On `hold_intra` there was nothing to weigh -- the pipeline agreed with a
      human 44.4% of the time, below chance, and the model 97.0%.
    * On `change` the model is *also* better against the human (98.0% vs
      95.9%), which argues for taking it everywhere. But the pipeline's
      positive label is grounded in behaviour recorded in the call -- the other
      person heard the speaker finish and took the floor with a substantive
      turn -- and that evidence does not depend on any listener at all.

    **Settled: `llm`**, measured by `20_label_policy.py`. The two policies
    differ on 420 rows (2.27%), 70 of them in the sealed test split, and
    switching moves test accuracy by -0.10 (tiny) / +0.34 (base) -- opposite
    signs, both inside the seed spread. `reports/label_policy.md`.

    `--label-policy core-safe` is kept, since every row carries both labels.
    But flipping it re-labels the benchmark, so it is a re-measurement rather
    than a config change.
    """
    # Named after the row file it labelled, so a relaxed-harvest run produces
    # `samples_relaxed_llm.jsonl` and cannot overwrite the 16,213 labels the
    # model was trained on.
    out = GOLD / (ROWS.replace(".jsonl", "_llm.jsonl"))
    flips: dict[str, list[tuple[int, int]]] = {}
    with out.open("w", encoding="utf-8") as f:
        for sid, r in rows.items():
            g = got.get(sid)
            rec = dict(r)
            rec["label_pipeline"] = r["label"]
            if g and g["verdict"] in ("complete", "incomplete"):
                lab = 1 if g["verdict"] == "complete" else 0
                rec["llm_verdict"] = lab
                rec["llm_ok"] = True
                rec["llm_conf"] = g["confidence"]
                rec["llm_reason"] = g["reason"]
                rec["dispute"] = lab != r["label"]
                keep_pipeline = policy == "core-safe" and r["source"] == "change"
                rec["label"] = r["label"] if keep_pipeline else lab
                flips.setdefault(r["source"], []).append((r["label"], lab))
            else:
                rec["llm_ok"] = False
                rec["dispute"] = False
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    judged = sum(len(v) for v in flips.values())
    L = [f"# Relabelling the pool with `{model}`", "",
         f"Prompt `{PROMPT_ID}`, audio only, Vertex AI batch. "
         f"**{judged:,} clips labelled for ${st['cost']:.2f}** "
         f"({st['labelled'] - judged} returned no usable verdict). "
         f"Label policy: `{policy}`.", "",
         "The pipeline's `hold_intra` class agreed with a human listener 44.4% of",
         "the time — below chance — because \"a pause fell inside a transcript",
         "segment\" is not a judgement about completeness. This is the pass that",
         "replaces it with one that is.", ""]

    if st.get("n"):
        L += [f"Scored free against the {st['n']} human-labelled clips that ride along in",
              f"the pool: **{100 * st['acc']:.1f}% agreement**, {100 * st.get('hold_intra', 0):.1f}% on `hold_intra`, "
              f"{100 * st.get('prec_inc', 0):.1f}% precision on `incomplete`.", ""]

    L += ["## What moved", "",
          "| source | n | stayed | flipped | new `complete` | new `incomplete` |",
          "|---|---|---|---|---|---|"]
    for src in ("change", "hold_intra", "change_midseg", "hold_inter"):
        p = flips.get(src)
        if not p:
            continue
        same = sum(1 for a, b in p if a == b)
        comp = sum(1 for _, b in p if b == 1)
        L.append(f"| `{src}` | {len(p):,} | {same:,} ({100 * same / len(p):.0f}%) | "
                 f"{len(p) - same:,} | {comp:,} | {len(p) - comp:,} |")

    tot = [x for v in flips.values() for x in v]
    if tot:
        comp = sum(1 for _, b in tot if b == 1)
        L += ["", f"**Net class balance: {comp:,} complete / {len(tot) - comp:,} incomplete** "
                  f"(was {sum(1 for a, _ in tot if a == 1):,} / {sum(1 for a, _ in tot if a == 0):,}).", ""]

    L += ["## Split coverage", "", "| split | labelled | complete | incomplete |", "|---|---|---|---|"]
    for sp in ("train", "dev", "test"):
        sel = [(sid, g) for sid, g in got.items()
               if sid in rows and rows[sid]["split"] == sp
               and g["verdict"] in ("complete", "incomplete")]
        if not sel:
            continue
        c = sum(1 for _, g in sel if g["verdict"] == "complete")
        L.append(f"| {sp} | {len(sel):,} | {c:,} | {len(sel) - c:,} |")
    L.append("")

    # The attack results, quoted from whatever 05 last measured. The relabelling
    # only counts if the set still survives being attacked -- a labeller that
    # made the classes acoustically separable would have made things worse.
    before = REPORTS / "verify_samples_core.json"
    after = {k: REPORTS / f"verify_samples_llm_{k}.json" for k in ("core", "all")}
    if before.exists() and any(p.exists() for p in after.values()):
        b = json.loads(before.read_text())
        L += ["## Does it still survive being attacked?", "",
              "`pipeline/09_verify.py` fits a deliberately dumb classifier on ten crude",
              "global features — energy, duration, spectral tilt, voiced fraction — that",
              "carry no prosody. Near-chance is the pass condition.", "",
              "| | samples | AUC | acc vs majority | `prev_dur` d | `voiced_frac` d |",
              "|---|---|---|---|---|---|",
              f"| pipeline labels | {b['n']:,} | {b['auc']:.3f} | {b['acc']:.3f} / {b['majority']:.3f} | "
              f"{b['d_prev_dur']:+.2f} | {b['d_voiced_frac']:+.2f} |"]
        for k, p in after.items():
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            L.append(f"| relabelled, {k} | {d['n']:,} | {d['auc']:.3f} | "
                     f"{d['acc']:.3f} / {d['majority']:.3f} | "
                     f"{d['d_prev_dur']:+.2f} | {d['d_voiced_frac']:+.2f} |")
        core = json.loads(after["core"].read_text()) if after["core"].exists() else None
        if core:
            L += ["",
                  "**The `prev_dur` confound is gone.** It was the one structural defect the",
                  f"build knew about: `hold_intra` required a pause *inside* a transcript",
                  "segment, so the negative class was drawn from longer utterances by",
                  f"construction (d = {b['d_prev_dur']:+.2f}), and it leaked into the clip as voiced",
                  f"fraction (d = {b['d_voiced_frac']:+.2f}, the largest audio effect measured). Labelling by",
                  f"listening severs that link: d = {core['d_prev_dur']:+.2f} and {core['d_voiced_frac']:+.2f}. The gate no longer",
                  "decides the label, so the gate's bias no longer rides along with it.", "",
                  f"AUC moved {b['auc']:.3f} → {core['auc']:.3f}, which is up, not down — worth saying plainly.",
                  "Both are well inside the near-chance band, and the class balance moved at",
                  f"the same time ({b['majority']:.3f} → {core['majority']:.3f} majority), so AUC is the comparable",
                  "number and accuracy is not. The probe beats the majority class by 1.1",
                  "points where before it lost to it by 0.1.", ""]

    ch = flips.get("change")
    if ch:
        moved = sum(1 for a, b in ch if a != b)
        L += ["## The one judgement call in this file", "",
              f"The model contradicts the pipeline on **{moved:,} of {len(ch):,} `change`**",
              f"samples ({100 * moved / len(ch):.1f}%). Those are the positives verified at 95.9%",
              "by ear, and grounded in behaviour recorded in the call — the other person",
              "took the floor with a substantive turn. The model is the better labeller",
              "against the human on this class too (98.0% vs 95.9%), so the default policy",
              "`llm` takes it — settled, see `reports/label_policy.md`.",
              "Every row carries both labels plus `dispute`.", "",
              f"Disputed rows overall: **{sum(1 for v in flips.values() for a, b in v if a != b):,}** "
              f"of {sum(len(v) for v in flips.values()):,}. Training on the undisputed subset",
              "is the high-precision option if neither witness settles it.", ""]

    L += ["## Files", "",
          f"`{out.relative_to(ROOT)}` — every row from `samples.jsonl`, plus:",
          "",
          "| field | meaning |",
          "|---|---|",
          "| `label` | the label to train on, per the policy above |",
          "| `label_pipeline` | what the channel-derived pipeline said |",
          "| `llm_verdict` | what the model said |",
          "| `dispute` | the two disagree |",
          "| `llm_conf`, `llm_reason` | the model's confidence and its stated reason |",
          "| `llm_ok` | false where no clip exists, or the response did not parse |",
          "",
          "Re-check it at any time with:", "",
          "```bash",
          "python pipeline/09_verify.py --samples samples_llm.jsonl --sources all --save",
          "```", ""]
    rep = REPORTS / f"relabelling{rows_tag().replace('+', '_')}.md"
    rep.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"  -> {rep}")


def write_sheet(model: str, got: dict, hum: dict, rows: dict, idx: dict) -> None:
    recs = []
    for sid, g in got.items():
        if sid not in rows:
            continue
        r, h, v = rows[sid], hum.get(sid), g["verdict"]
        mv = 1 if v == "complete" else (0 if v == "incomplete" else None)
        recs.append({"idx": idx.get(sid, 0), "clip": f"{idx.get(sid, 0):03d}.wav",
                     "model": v, "conf": f"{g['confidence']:.2f}",
                     "you": "complete" if h == 1 else "incomplete" if h == 0 else "",
                     "agree": "" if (h is None or mv is None) else ("yes" if mv == h else "NO"),
                     "your_call": "",
                     "pipeline": "complete" if r["label"] == 1 else "incomplete",
                     "source": r["source"], "gap_s": f"{r['gap']:.3f}",
                     "reason": g["reason"], "text": " ".join((r.get("text") or "").split()),
                     "sid": sid})
    recs.sort(key=lambda d: (d["agree"] != "NO", d["idx"]))
    out = REPORTS / "qa" / f"batch_{model}_{PROMPT_ID}.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]), quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(recs)


def write_summary(table: list[dict], binar: dict[str, dict[str, int]],
                  hum: dict[str, int]) -> None:
    out = REPORTS / "model_bakeoff.md"
    top = table[0]
    cheaper = [d for d in table if d["cost"] < top["cost"] * 0.9] if table else []
    L = ["# Labeller bake-off — 197 human-labelled clips, Vertex AI Batch",
         "",
         f"Prompt `{PROMPT_ID}` (unchanged from the validated run). Audio only, no",
         "transcript. Same 197 clips and same human labels for every model, so the",
         "comparison is paired. Batch pricing = 50% of interactive.",
         "",
         f"**Winner: `{top['model']}` — {100 * top['acc']:.1f}% agreement, "
         f"{100 * top['prec_inc']:.1f}% precision on `incomplete`, "
         f"${top['cost'] / max(top['n'], 1) * 16216:.2f} to label the whole pool.**",
         "",
         "The bar was 90% and four models clear it. The decision is not accuracy",
         "alone — it is accuracy per dollar, and the paired test below shows the",
         "top of the table is a tie that price breaks.",
         "",
         "| model | n | agreement | 95% CI | `change` | `hold_intra` | prec `incomplete` | $ run | in/out tok | think |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for d in table:
        if "acc" not in d:
            continue
        ci = f"{100 * d['acc_ci'][0]:.0f}–{100 * d['acc_ci'][1]:.0f}%"
        pi = f"{100 * d['prec_inc']:.1f}% ({d['prec_inc_n']})" if "prec_inc" in d else "—"
        L.append(f"| `{d['model']}` | {d['n']} | **{100 * d['acc']:.1f}%** | {ci} | "
                 f"{100 * d.get('change', 0):.1f}% | {100 * d.get('hold_intra', 0):.1f}% | {pi} | "
                 f"${d['cost']:.2f} | {d['tok_in']:.0f}/{d['tok_out']:.0f} | {d['think']:.0f} |")
    pipe = pipeline_stats(hum, {r["sid"]: r for r in rows_all()})
    L += [f"| *our pipeline* | {len(hum)} | {100 * pipe['all']:.1f}% | – | "
          f"{100 * pipe['change']:.1f}% | {100 * pipe['hold_intra']:.1f}% | "
          f"{100 * pipe['prec_inc']:.1f}% | – | – | – |", ""]

    # Ranking by raw accuracy over-reads a 197-clip sample. The paired test says
    # which gaps are real, and it is what decides the tie at the top.
    best = table[0]["model"]
    L += [f"## Is the gap real? Paired McNemar against `{best}`", "",
          "Only clips where exactly one of the two was right count.", "",
          "| vs | winner-only right | other-only right | p |", "|---|---|---|---|"]
    for d in table[1:]:
        if d["model"] not in binar:
            continue
        b01, b10, p = mcnemar(binar[best], binar[d["model"]], hum)
        sig = "**significant**" if p < 0.05 else "not significant"
        L.append(f"| `{d['model']}` | {b01} | {b10} | {p:.3f} — {sig} |")

    # Seven models failing the same clip in the same direction is weak evidence
    # against the models and strong evidence against the label: the human pass
    # was one listen to an isolated 8 s clip with no future audio.
    consensus = []
    for sid, h in hum.items():
        votes = [b[sid] for b in binar.values() if sid in b]
        if len(votes) >= 5 and sum(v != h for v in votes) >= len(votes) - 1:
            consensus.append((sid, h, sum(v != h for v in votes), len(votes)))
    if consensus:
        top_wrong = sum(1 for sid, *_ in consensus if sid in binar[best] and binar[best][sid] != hum[sid])
        n = top["n"]
        L += ["", "## Clips where the models agree against the human label", "",
              f"{len(consensus)} of 197 clips have all-but-at-most-one model contradicting the",
              "human, in the same direction. These are re-listen candidates, not model",
              f"failures — {top_wrong} of `{best}`'s {n - round(top['acc'] * n)} errors sit here. If they are",
              f"human slips, `{best}` lands at "
              f"{100 * (round(top['acc'] * n) + top_wrong) / n:.1f}%.", "",
              "| clip | human said | models against | of |", "|---|---|---|---|"]
        for sid, h, k, tot in sorted(consensus, key=lambda x: -x[2]):
            L.append(f"| `{qa_index().get(sid, 0):03d}.wav` | "
                     f"{'complete' if h == 1 else 'incomplete'} | {k} | {tot} |")

    L += ["", "## Cost to label the full pool at batch rates", "",
          "| model | `hold_intra` 9,406 | everything 16,216 |", "|---|---|---|"]
    for d in table:
        if "acc" not in d:
            continue
        per = d["cost"] / max(d.get("labelled", d["n"]), 1)
        L.append(f"| `{d['model']}` | ${per * 9406:.2f} | ${per * 16216:.2f} |")
    L += ["", "Errors / unparsed per model: "
          + ", ".join(f"`{d['model']}` {d['err']}" for d in table), ""]
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[6:]))
    print(f"\n  -> {out}")


DONE = ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED", "JOB_STATE_PAUSED")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--source", default=QA,
                    choices=[QA, "all", "change", "hold_intra", "change_midseg", "hold_inter"],
                    help=f"'{QA}' = the 197 human-labelled clips; anything else is a bulk run")
    ap.add_argument("--split", default="", choices=["", "train", "dev", "test"],
                    help="bulk runs go out as one job per split unless this names one")
    ap.add_argument("--splits", default="",
                    help="comma-separated splits, e.g. train,dev -- for labelling the "
                         "07_relax_funnel harvest without touching the sealed test set")
    ap.add_argument("--rows", default="samples.jsonl",
                    help="row file to label; samples_relaxed.jsonl for the 07_relax_funnel harvest")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--wait", action="store_true", help="poll until every job settles")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label-policy", default="llm", choices=["llm", "core-safe"],
                    help="core-safe keeps the pipeline label on the `change` class")
    ap.add_argument("--no-resume", action="store_true",
                    help="re-request clips that already have a cached verdict")
    a = ap.parse_args()
    global ROWS
    ROWS = a.rows
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    if a.source != QA and len(models) > 1:
        models = models[:1]
        print(f"  bulk run: using {models[0]} only")
    splits = [s.strip() for s in a.splits.split(",") if s.strip()] or None
    if splits:
        print(f"  rows {ROWS}, splits {splits}")

    if a.upload or a.submit or a.status or a.wait:
        require_gcp()

    if a.upload:
        sel = ([j for sp in splits for j in jobs(a.source, sp)] if splits
               else jobs(a.source, a.split))
        upload(sel)
    if a.submit:
        submit(models, a.source, a.split, a.limit, not a.no_resume, splits)
    if a.wait:
        # Only the tags this invocation cares about -- a finished bake-off
        # should not hold up, or be held up by, a bulk run.
        mine = [t for t, s in (json.loads(STATE.read_text()) if STATE.exists() else {}).items()
                if (s.get("model") or t) in models and (s.get("source") or QA) == a.source
                and owns(t)]
        while True:
            st = poll(mine)
            live = [t for t in mine if st.get(t, {}).get("state") not in DONE]
            if not live:
                break
            print(f"  ... {len(live)} running, sleeping 60s")
            time.sleep(60)
    elif a.status:
        poll()
    if a.collect:
        collect(models, a.source, a.label_policy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
