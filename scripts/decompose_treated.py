"""Treated-subset and whole-store decompositions reported in paper section 6 (read-only).

    python -m scripts.decompose_treated

Two analyses requested in internal review, both computable
from existing artifacts with zero model calls:

  A. Mem0: on the turns where it actually injected, what does it deliver? (distinguishes
     "extraction stores nothing" from "extraction stores something worse")
  B. A-Mem: on how many judged turns did top-k return the entire store, and what is
     A-Mem - B1 restricted to turns where selection was actually exercised (store > k)?
"""
import json, math, os, sys
from collections import defaultdict

H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(H, "src"))
os.chdir(H)
from harness.registry import load_dialogues  # noqa: E402

S = json.load(open("runs/stats_summary.json", encoding="utf-8"))

def rid(coll, cond, abl=None, k=5):
    for r, v in S.items():
        if (v.get("collection") == coll and v.get("condition") == cond
                and v.get("context") == "memory_only" and v.get("memory_k") == k
                and (v.get("ablation") or {}) == (abl or {})):
            return r
    return None

def rewrites(r):
    p = f"runs/rewrites/{r}.jsonl"
    return {j["turn_id"]: j for j in (json.loads(l) for l in open(p, encoding="utf-8")) }

def per_turn(r):
    p = f"runs/metrics/{r}.per_turn.jsonl"
    return {j["turn_id"]: j["ndcg_cut_3"] for j in (json.loads(l) for l in open(p, encoding="utf-8"))}

def paired(a, b, ids):
    d = [b[t] - a[t] for t in ids]
    n = len(d); m = sum(d)/n if n else 0.0
    if n < 2: return m, n, 1.0
    sd = math.sqrt(sum((x-m)**2 for x in d)/(n-1))
    if sd == 0: return m, n, 1.0
    t = m/(sd/math.sqrt(n))
    try:
        from scipy import stats as sp
        p = float(sp.t.sf(abs(t), n-1)*2)
    except Exception: p = float("nan")
    return m, n, p

def wilcoxon(a, b, ids):
    d = [b[t]-a[t] for t in ids if b[t] != a[t]]
    if len(d) < 5: return float("nan")
    from scipy import stats as sp
    return float(sp.wilcoxon(d).pvalue)

for coll in ("cast19", "cast20"):
    dlg = {d.dialogue_id: d for d in load_dialogues(coll, data_dir="data")}
    # prior ingested turns (user+system alike -- replay ingests every turn) per turn_id
    prior = {}
    for d in dlg.values():
        for i, t in enumerate(d.turns):
            prior[t.turn_id] = i
    b0, b1 = rid(coll, "B0_none"), rid(coll, "B1_full_history")
    s1, s3 = rid(coll, "S1_mem0"), rid(coll, "S3_amem")
    s1n = rid(coll, "S1_mem0", {"infer": False})
    RW = {c: rewrites(r) for c, r in (("B0", b0), ("S1", s1), ("S3", s3)) if r}
    PT = {c: per_turn(r) for c, r in (("B0", b0), ("B1", b1), ("S1", s1), ("S3", s3),
                                      ("S1n", s1n)) if r}
    judged = sorted(set(PT["B0"]) & set(PT["B1"]))
    print(f"\n================ {coll} (judged n={len(judged)}) ================")

    # ---- A. Mem0 treated-subset decomposition
    treated = [t for t in judged if RW["S1"][t]["query"].strip() != RW["B0"][t]["query"].strip()]
    untr = [t for t in judged if t not in treated]
    hb, nb, pb = paired(PT["B0"], PT["B1"], treated)      # headroom on treated turns
    hs, ns, ps = paired(PT["B0"], PT["S1"], treated)      # Mem0 gain on treated turns
    print(f"A. Mem0 injected+changed on {len(treated)}/{len(judged)} judged turns "
          f"({100*len(treated)/len(judged):.0f}%)")
    print(f"   on TREATED turns: B1-B0 = {hb:+.4f} | S1-B0 = {hs:+.4f} "
          f"(wilcoxon p={wilcoxon(PT['B0'],PT['S1'],treated):.2e})")
    if hb: print(f"   => delivered share ON TREATED TURNS: {100*hs/hb:.1f}%")
    hu, nu, _ = paired(PT["B0"], PT["B1"], untr)
    print(f"   on UNTREATED turns ({nu}): B1-B0 = {hu:+.4f} (the gain Mem0 left on the table)")
    if s1n:
        hn, _, _ = paired(PT["B0"], PT["S1n"], treated)
        print(f"   S1[infer=False]-B0 on the same treated subset: {hn:+.4f}")

    # ---- B. A-Mem whole-store vs selection-active turns (k=5)
    k = 5
    whole = [t for t in judged if prior[t] <= k]
    sel   = [t for t in judged if prior[t] > k]
    print(f"B. A-Mem k={k}: store<=k (whole store returned) on {len(whole)}/{len(judged)} "
          f"judged turns ({100*len(whole)/len(judged):.0f}%)")
    for name, sub in (("whole-store", whole), ("selection-active (store>k)", sel)):
        m, n, p = paired(PT["B1"], PT["S3"], sub)
        print(f"   S3-B1 on {name:28s} n={n:3d}: {m:+.4f} (t-test p={p:.3f})")
    mh, nh, _ = paired(PT["B0"], PT["B1"], sel)
    m3, _, _ = paired(PT["B0"], PT["S3"], sel)
    if mh: print(f"   on selection-active turns: headroom {mh:+.4f}, S3 delivers {100*m3/mh:.0f}%")
