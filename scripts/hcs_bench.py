"""Real-scoresheet benchmark from HCS (E29): real written entries + simulated gaps.

Truth = the legal game recovered by scripts/hcs_reconstruct.py. Kept: standard start, >= 30 plies, recovered
with at most MAX_RATE of entries differing (so the recovered truth is trustworthy); one sheet per game (the
copy needing fewer edits); the 'test' label copies are skipped (they repeat training-label games).
Gaps: the 'missing' preset's time-pressure bursts. Two presets on identical gaps:
  hcs_real   tokens carry what the player wrote (real slips, abbreviations, blanks)
  hcs_clean  tokens carry the true SAN (isolates the effect of real slips)
"""
import json, random, sys
import chess
from dataclasses import replace
from recon.bench.build import _h, make_instance
from recon.game import Game
from recon.observe.corrupt import PRESETS

MAX_RATE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
rows = [json.loads(l) for l in open("data/hcs/recon.jsonl")]
best = {}
for r in rows:
    if r["split"] != "train" or r["n"] < 30 or r["n_diff"] / r["n"] > MAX_RATE:
        continue
    if r["game"] not in best or r["n_diff"] < best[r["game"]]["n_diff"]:
        best[r["game"]] = r

out = []
for g, r in sorted(best.items()):
    b = chess.Board(); uci = []
    for s in r["truth"]:
        m = b.parse_san(s); uci.append(m.uci()); b.push(m)
    game = Game(id=f"hcs{g}", moves=uci)
    inst = make_instance(game, PRESETS["missing"], seed=0, split="hcs")
    src, tsrc = inst["input"]["sources"][0], inst["truth"]["sources"][0]
    for preset in ("hcs_real", "hcs_clean"):
        x = json.loads(json.dumps(inst))
        toks, kinds, tply = [], [], []
        omitted = set(x["truth"]["sources"][0]["omitted_plies"])
        for tok, p in zip(x["input"]["sources"][0]["tokens"], x["truth"]["sources"][0]["token_ply"]):
            text = r["written"][p] if preset == "hcs_real" else r["truth"][p]
            if not text:  # a box the player left blank: no token, the ply joins the gap
                omitted.add(p); continue
            tok["text"] = text
            toks.append(tok); tply.append(p)
            kinds.append("match" if text.replace("+", "").replace("#", "").replace("0", "O") == r["truth"][p] else "slip")
        x["input"]["sources"][0]["tokens"] = toks
        ts = x["truth"]["sources"][0]
        ts["token_ply"], ts["token_kind"], ts["omitted_plies"] = tply, kinds, sorted(omitted)
        x["truth"]["hidden_plies"] = sorted(omitted)
        x["id"], x["preset"] = f"{preset}/hcs{g}", preset
        x["meta"].update(num_hidden=len(omitted), num_token_errors=sum(k != "match" for k in kinds),
                         rating_band="unknown", speed="otb", hcs_sheet=r["id"])
        x["input"]["players"] = {"white": {"rating": None, "rating_system": None}, "black": {"rating": None, "rating_system": None}}
        out.append(x)
with open("bench/hcs.jsonl", "w") as f:
    for x in out:
        f.write(json.dumps(x) + "\n")
real = [x for x in out if x["preset"] == "hcs_real"]
print(f"games {len(real)}  plies {sum(x['meta']['num_plies'] for x in real)}  hidden {sum(x['meta']['num_hidden'] for x in real)}  "
      f"wrong written tokens {sum(x['meta']['num_token_errors'] for x in real)} / {sum(len(x['input']['sources'][0]['tokens']) for x in real)}")
