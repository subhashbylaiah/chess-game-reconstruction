"""95% confidence intervals for top-k from per-ply rank files, resampling whole games (plies in a game are
correlated; all presets of a game are resampled together). Usage: bootstrap_ci.py ranks.jsonl [more ...]"""
import json, random, sys, collections

B = 2000
for path in sys.argv[1:]:
    games = collections.defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        games[r["id"].split("/", 1)[-1]].append(r["rank"])
    keys = list(games)
    def stats(ks):
        rs = [x for k in ks for x in games[k]]
        n = len(rs)
        return [sum(x <= k for x in rs) / n for k in (1, 2, 5)]
    point = stats(keys)
    rng = random.Random(0)
    boot = [stats([rng.choice(keys) for _ in keys]) for _ in range(B)]
    ci = []
    for j in range(3):
        v = sorted(b[j] for b in boot)
        ci.append((v[int(0.025 * B)], v[int(0.975 * B)]))
    print(f"{path}: games {len(keys)} " + "  ".join(
        f"top{k} {100*p:.1f} [{100*lo:.1f}, {100*hi:.1f}]" for k, p, (lo, hi) in zip((1, 2, 5), point, ci)))
