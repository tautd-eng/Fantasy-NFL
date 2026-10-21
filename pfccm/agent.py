#!/usr/bin/env python3
"""PFCCM (NeonSportz, Madden 26 coach mode) agentas.

1. Parsisiunčia lygos duomenis iš NeonSportz API (komandos, žaidėjai, rungtynės, statistika).
2. Archyvuoja kopiją (pfccm/data/archive/<data>/*.csv.gz), jei duomenys pasikeitė nuo paskutinės kopijos.
3. Skaičiuoja analizę -> pfccm/data/site.json, kurį skaito pfccm/index.html:
   - normalizuoti rodikliai (per bandymą / dalis komandos totalo), OP kandidatai (virš OVR, konteksto, algos)
   - atributų (įsk. ūgį, svorį) poveikis produkcijai pagal poziciją; OL – komandų lygiu
   - mano komandos (config.json -> team) žaidėjų įvertinimas ir pozicijų rangai
"""
import csv, datetime, gzip, hashlib, io, json, os, sys, time, urllib.request
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
CFG = json.load(open(os.path.join(ROOT, "config.json")))
LEAGUE = CFG.get("league", "PFCCM")
MYTEAM = CFG.get("team", "HOU")
BASE = f"https://neonsportz.com/api/leagues/{LEAGUE}"
UA = {"User-Agent": "Mozilla/5.0 (pfccm-dashboard; github-actions)", "Accept": "application/json"}


# ---------------------------------------------------------------- fetch
def get_json(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))


def get_all(path):
    url = BASE + path + ("&" if "?" in path else "?") + "size=100"
    out, seen, guard = [], set(), 0
    while url and guard < 300:
        guard += 1
        j = get_json(url)
        if isinstance(j, list):
            return j
        out += j.get("results", [])
        url = j.get("next")
        if url in seen:
            break
        seen.add(url)
    return out


def fetch_all():
    teams = get_all("/teams/?size=0") or get_all("/teams/")
    players = get_all("/players/")
    games = get_all("/games/")
    season = max(int(t.get("seasonIndex") or 0) for t in teams)
    stats = {}
    for cat in ["passing", "rushing", "receiving", "defense", "kicking", "punting"]:
        for stage in (0, 1, 2):
            rows = get_all(f"/stats/{cat}/?seasonIndex={season}&stageIndex={stage}")
            seen, uniq = set(), []
            for r in rows:
                k = (r.get("player__rosterId"), r.get("team__abbrName"), r.get("seasonIndex"), r.get("stageIndex"))
                if k not in seen:
                    seen.add(k)
                    uniq.append(r)
            if uniq:
                stats[(cat, stage)] = uniq
    gseen, guniq = set(), []
    for g in games:
        if g.get("pk") not in gseen:
            gseen.add(g.get("pk"))
            guniq.append(g)
    return teams, players, guniq, stats, season


# ---------------------------------------------------------------- archive
def flat(o):
    r = {}
    for k, v in o.items():
        if v is None:
            r[k] = ""
        elif not isinstance(v, (dict, list)):
            r[k] = v
        elif k in ("team", "homeTeam", "awayTeam") and isinstance(v, dict):
            r[{"team": "teamAbbr", "homeTeam": "homeAbbr", "awayTeam": "awayAbbr"}[k]] = v.get("abbrName", "")
        elif k == "leaguemember" and isinstance(v, dict):
            r["coachUser"] = (v.get("user") or {}).get("username", "")
        elif isinstance(v, list) and all(not isinstance(x, (dict, list)) for x in v):
            r[k] = "|".join(map(str, v))
    return r


def to_csv(rows):
    F = [flat(r) for r in rows]
    cols = []
    for r in F:
        for c in r:
            if c not in cols:
                cols.append(c)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(F)
    return buf.getvalue()


def archive(teams, players, games, stats, season):
    files = {f"players.csv": to_csv(players), "teams.csv": to_csv(teams), "games.csv": to_csv(games)}
    for cat in ["passing", "rushing", "receiving", "defense", "kicking", "punting"]:
        rows = []
        for st in (0, 1, 2):
            for r in stats.get((cat, st), []):
                rows.append({"snapStage": st, **r})
        if rows:
            files[f"stats_{cat}.csv"] = to_csv(rows)
    # pokyčio patikra: ignoruoti last_updated tipo laukus
    sig = hashlib.sha256()
    for name in sorted(files):
        for line in files[name].splitlines():
            sig.update(line.encode())
    digest = sig.hexdigest()
    os.makedirs(os.path.join(DATA, "archive"), exist_ok=True)
    idx_path = os.path.join(DATA, "archive", "index.json")
    idx = json.load(open(idx_path)) if os.path.exists(idx_path) else []
    if idx and idx[-1].get("hash") == digest:
        return idx, False
    day = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    d = os.path.join(DATA, "archive", day)
    os.makedirs(d, exist_ok=True)
    meta = []
    for name, text in files.items():
        with gzip.open(os.path.join(d, name + ".gz"), "wt", encoding="utf-8") as f:
            f.write(text)
        meta.append({"file": name + ".gz", "rows": max(0, text.count("\n") - 1)})
    idx = [e for e in idx if e["date"] != day] + [{"date": day, "season": season, "hash": digest, "files": meta}]
    json.dump(idx, open(idx_path, "w"), indent=1)
    return idx, True


# ---------------------------------------------------------------- math
def ridge(X, y, lam):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Z = (X - mu) / sd
    ym = y.mean()
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return w, (lambda Xn: ym + ((np.asarray(Xn, float) - mu) / sd) @ w)


def cv_r2(X, y, lam, k=5):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n = len(y)
    if n < 10:
        return float("nan")
    idx = np.argsort((np.arange(n) * 7919) % n)
    pred = np.zeros(n)
    for f in range(k):
        te = idx[f::k]
        tr = np.setdiff1d(idx, te)
        _, p = ridge(X[tr], y[tr], lam)
        pred[te] = p(X[te])
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def z(a):
    a = np.asarray(a, float)
    s = a.std()
    return (a - a.mean()) / (s if s else 1)


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def r(x, d=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


# ---------------------------------------------------------------- analysis
GRP = {"QB": "QB", "HB": "HB", "WR": "WR", "TE": "TE", "LEDGE": "DL", "REDGE": "DL", "DT": "DL",
       "MIKE": "LB", "WILL": "LB", "SAM": "LB", "CB": "DB", "FS": "DB", "SS": "DB",
       "LT": "OL", "LG": "OL", "C": "OL", "RG": "OL", "RT": "OL", "K": "K", "P": "P"}
DEV = ["N", "Star", "SS", "XF"]
ATT = {
    "QB": "aware,throwAccShort,throwAccMid,throwAccDeep,throwPower,throwUnderPressure,throwOnRun,playAction,breakSack,speed,agility",
    "HB": "speed,accel,agility,changeOfDirection,breakTackle,bCV,jukeMove,spinMove,stiffArm,truck,carry,aware,catch,strength",
    "WR": "speed,accel,agility,changeOfDirection,release,routeRunShort,routeRunMed,routeRunDeep,catch,cIT,specCatch,jump,aware",
    "TE": "speed,release,routeRunShort,routeRunMed,routeRunDeep,catch,cIT,specCatch,aware,strength,breakTackle,jump",
    "DL": "speed,accel,agility,strength,powerMoves,finesseMoves,blockShed,pursuit,tackle,playRec,aware,hitPower",
    "LB": "speed,accel,playRec,zoneCover,manCover,tackle,pursuit,blockShed,hitPower,aware,powerMoves,finesseMoves",
    "DB": "speed,accel,agility,changeOfDirection,manCover,zoneCover,press,playRec,catch,aware,jump,tackle,pursuit",
}
OL_ATT = "passBlock,passBlockPower,passBlockFinesse,runBlock,runBlockPower,runBlockFinesse,impactBlock,strength,aware,agility".split(",")
NAMES = {"aware": "Awareness", "throwAccShort": "Short Acc", "throwAccMid": "Mid Acc", "throwAccDeep": "Deep Acc", "throwPower": "Throw Power",
         "throwUnderPressure": "Under Pressure", "throwOnRun": "On the Run", "playAction": "Play Action", "breakSack": "Break Sack",
         "speed": "Speed", "agility": "Agility", "accel": "Acceleration", "changeOfDirection": "Change of Dir.", "breakTackle": "Break Tackle",
         "bCV": "BC Vision", "jukeMove": "Juke", "spinMove": "Spin", "stiffArm": "Stiff Arm", "truck": "Trucking", "carry": "Carrying",
         "catch": "Catching", "strength": "Strength", "release": "Release", "routeRunShort": "Short Route", "routeRunMed": "Medium Route",
         "routeRunDeep": "Deep Route", "cIT": "Catch in Traffic", "specCatch": "Spectacular Catch", "jump": "Jumping", "powerMoves": "Power Moves",
         "finesseMoves": "Finesse Moves", "blockShed": "Block Shedding", "pursuit": "Pursuit", "tackle": "Tackle", "playRec": "Play Recognition",
         "hitPower": "Hit Power", "zoneCover": "Zone Coverage", "manCover": "Man Coverage", "press": "Press", "height": "Ūgis (col.)",
         "weight": "Svoris (lb)", "passBlock": "Pass Block", "passBlockPower": "Pass Block Power", "passBlockFinesse": "Pass Block Finesse",
         "runBlock": "Run Block", "runBlockPower": "Run Block Power", "runBlockFinesse": "Run Block Finesse", "impactBlock": "Impact Block",
         "kickPower": "Kick Power", "kickAcc": "Kick Accuracy"}
KEYATT = {"QB": ["aware", "throwAccMid"], "HB": ["breakTackle", "spinMove"], "WR": ["speed", "release"], "TE": ["speed", "catch"],
          "DL": ["speed", "playRec"], "LB": ["playRec", "zoneCover"], "DB": ["aware", "pursuit"], "OL": ["aware", "passBlockPower"],
          "K": ["kickPower", "kickAcc"], "P": ["kickPower", "kickAcc"]}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def att(p, a):
    return f(p.get(a)) if a in ("height", "weight") else f(p.get(a + "Rating"))


def analyze(teams, players, stats):
    P = [p for p in players if not p.get("isRetired")]
    byid = {str(p["rosterId"]): p for p in P}
    abbr = lambda p: ((p.get("team") or {}).get("abbrName") if isinstance(p.get("team"), dict) else None)
    reg = {cat: stats.get((cat, 1), []) for cat in ["passing", "rushing", "receiving", "defense"]}
    # vienas eilutė žaidėjui: daugiausiai rungtynių
    st = {}
    for cat, rows in reg.items():
        m = {}
        for s in rows:
            k = str(s.get("player__rosterId"))
            if k not in m or f(s.get("gamesPlayed")) > f(m[k].get("gamesPlayed")):
                m[k] = s
        st[cat] = m
    tnames = sorted({t["abbrName"] for t in teams})

    def tsum(cat, t, k):
        return sum(f(s.get(k)) for s in reg[cat] if s.get("team__abbrName") == t)

    TT = {}
    tinfo = {t["abbrName"]: t for t in teams}
    for t in tnames:
        TT[t] = dict(passAtt=tsum("passing", t, "passTotalAtt"), sacksTaken=tsum("passing", t, "passTotalSacks"),
                     rushAtt=tsum("rushing", t, "rushTotalAtt"), recYds=tsum("receiving", t, "recTotalYds"),
                     tackles=tsum("defense", t, "defTotalTackles"), sacks=tsum("defense", t, "defTotalSacks"),
                     ints=tsum("defense", t, "defTotalInts"),
                     ballsDef=tsum("defense", t, "defTotalInts") + tsum("defense", t, "defTotalDeflections"),
                     defPassYds=f(tinfo[t].get("defPassYds")))

    # komandos kontekstas iš dabartinių sudėčių
    def top(t, poss, n, key=lambda p: f(p.get("playerBestOvr"))):
        l = sorted([p for p in P if abbr(p) == t and p.get("position") in poss and not p.get("isFreeAgent")], key=key, reverse=True)[:n]
        return l

    CTX = {}
    for t in tnames:
        ol = [x for pos in ["LT", "LG", "C", "RG", "RT"] for x in top(t, [pos], 1)]
        avg = lambda l, k: float(np.mean([f(p.get(k)) for p in l])) if l else 0.0
        CTX[t] = dict(qbOvr=avg(top(t, ["QB"], 1), "playerBestOvr"), olPB=avg(ol, "passBlockRating"), olRB=avg(ol, "runBlockRating"),
                      rcOvr=avg(top(t, ["WR", "TE"], 3), "playerBestOvr"),
                      dlOvr=avg(top(t, ["LEDGE", "REDGE", "DT"], 4), "playerBestOvr"),
                      lbOvr=avg(top(t, ["MIKE", "WILL", "SAM"], 3), "playerBestOvr"),
                      dbOvr=avg(top(t, ["CB", "FS", "SS"], 5), "playerBestOvr"), ol=ol)
    CTXK = {"QB": ["olPB", "rcOvr"], "HB": ["olRB", "qbOvr"], "WR": ["qbOvr", "olPB"], "TE": ["qbOvr", "olPB"],
            "DL": ["dbOvr", "lbOvr"], "LB": ["dlOvr", "dbOvr"], "DB": ["dlOvr", "lbOvr"]}

    def team_of(pid):
        for cat in ["passing", "rushing", "receiving", "defense"]:
            if pid in st[cat]:
                return st[cat][pid].get("team__abbrName")
        return None

    def sd(pid, cat):
        return st[cat].get(pid)

    def g(s, k):
        return f(s.get(k)) if s else 0.0

    def safe(a, b):
        return a / b if b else float("nan")

    M = {
        "QB": (lambda i: g(sd(i, "passing"), "passTotalAtt") >= 150, {
            "ANY/A": lambda i: (lambda s: safe(g(s, "passTotalYds") + 20 * g(s, "passTotalTDs") - 45 * g(s, "passTotalInts"), g(s, "passTotalAtt") + g(s, "passTotalSacks")))(sd(i, "passing")),
            "Comp%": lambda i: (lambda s: 100 * safe(g(s, "passTotalComp"), g(s, "passTotalAtt")))(sd(i, "passing")),
            "Sack% (−)": lambda i: (lambda s: -100 * safe(g(s, "passTotalSacks"), g(s, "passTotalAtt") + g(s, "passTotalSacks")))(sd(i, "passing"))}),
        "HB": (lambda i: g(sd(i, "rushing"), "rushTotalAtt") >= 50, {
            "Yds/touch": lambda i: safe(g(sd(i, "rushing"), "rushTotalYds") + g(sd(i, "receiving"), "recTotalYds"), g(sd(i, "rushing"), "rushTotalAtt") + g(sd(i, "receiving"), "recTotalCatches")),
            "BrokenTkl/att": lambda i: safe(g(sd(i, "rushing"), "rushTotalBrokenTackles"), g(sd(i, "rushing"), "rushTotalAtt")),
            "YAC/att": lambda i: safe(g(sd(i, "rushing"), "rushTotalYdsAfterContact"), g(sd(i, "rushing"), "rushTotalAtt")),
            "Carry share": lambda i: safe(g(sd(i, "rushing"), "rushTotalAtt"), TT[sd(i, "rushing")["team__abbrName"]]["rushAtt"])}),
        "WR": (lambda i: g(sd(i, "receiving"), "recTotalCatches") >= 20, {
            "RecYds share": lambda i: safe(g(sd(i, "receiving"), "recTotalYds"), TT[sd(i, "receiving")["team__abbrName"]]["recYds"]),
            "Yds/catch": lambda i: safe(g(sd(i, "receiving"), "recTotalYds"), g(sd(i, "receiving"), "recTotalCatches")),
            "YAC/catch": lambda i: g(sd(i, "receiving"), "recAvgYacPerCatch"),
            "Drops/100 (−)": lambda i: -100 * safe(g(sd(i, "receiving"), "recTotalDrops"), g(sd(i, "receiving"), "recTotalCatches"))}),
        "TE": (lambda i: g(sd(i, "receiving"), "recTotalCatches") >= 20, {
            "RecYds share": lambda i: safe(g(sd(i, "receiving"), "recTotalYds"), TT[sd(i, "receiving")["team__abbrName"]]["recYds"]),
            "Yds/catch": lambda i: safe(g(sd(i, "receiving"), "recTotalYds"), g(sd(i, "receiving"), "recTotalCatches")),
            "YAC/catch": lambda i: g(sd(i, "receiving"), "recAvgYacPerCatch")}),
        "DL": (lambda i: g(sd(i, "defense"), "gamesPlayed") >= 8, {
            "Sack share": lambda i: safe(g(sd(i, "defense"), "defTotalSacks"), TT[sd(i, "defense")["team__abbrName"]]["sacks"]),
            "Sacks/G": lambda i: safe(g(sd(i, "defense"), "defTotalSacks"), g(sd(i, "defense"), "gamesPlayed")),
            "Tackle share": lambda i: safe(g(sd(i, "defense"), "defTotalTackles"), TT[sd(i, "defense")["team__abbrName"]]["tackles"])}),
        "LB": (lambda i: g(sd(i, "defense"), "gamesPlayed") >= 8, {
            "Tackle share": lambda i: safe(g(sd(i, "defense"), "defTotalTackles"), TT[sd(i, "defense")["team__abbrName"]]["tackles"]),
            "Plays on ball share": lambda i: safe(g(sd(i, "defense"), "defTotalInts") + g(sd(i, "defense"), "defTotalDeflections"), TT[sd(i, "defense")["team__abbrName"]]["ballsDef"]),
            "Sack share": lambda i: safe(g(sd(i, "defense"), "defTotalSacks"), TT[sd(i, "defense")["team__abbrName"]]["sacks"])}),
        "DB": (lambda i: g(sd(i, "defense"), "gamesPlayed") >= 8, {
            "Plays on ball share": lambda i: safe(g(sd(i, "defense"), "defTotalInts") + g(sd(i, "defense"), "defTotalDeflections"), TT[sd(i, "defense")["team__abbrName"]]["ballsDef"]),
            "INT share": lambda i: safe(g(sd(i, "defense"), "defTotalInts"), max(TT[sd(i, "defense")["team__abbrName"]]["ints"], 1)),
            "Catches allowed/1k (−)": lambda i: -safe(g(sd(i, "defense"), "defTotalCatchAllowed"), TT[sd(i, "defense")["team__abbrName"]]["defPassYds"] / 1000),
            "Tackle share": lambda i: safe(g(sd(i, "defense"), "defTotalTackles"), TT[sd(i, "defense")["team__abbrName"]]["tackles"])}),
    }

    norm_rows, summary, attr_rows, metric_rows, tier_rows = [], [], [], [], []
    rng = np.random.default_rng(7)

    def attr_block(G, atts, X, comp, metrics):
        X = np.asarray(X, float)
        lam = max(2.0, len(comp) * 0.1)
        w, _ = ridge(X, comp, lam)
        cnt = np.zeros(len(atts))
        for _ in range(150):
            ix = rng.integers(0, len(comp), len(comp))
            wb, _ = ridge(X[ix], np.asarray(comp)[ix], lam)
            cnt[np.argsort(-wb)[:3]] += 1
        for j, a in enumerate(atts):
            attr_rows.append(dict(group=G, attribute=NAMES.get(a, a), corr=r(corr(X[:, j], comp)), weight=r(w[j], 3),
                                  top3Pct=int(round(100 * cnt[j] / 150)), leagueAvg=r(X[:, j].mean(), 1)))
        for name, y in metrics:
            wm, _ = ridge(X, y, lam)
            o = np.argsort(-wm)[:3]
            metric_rows.append(dict(group=G, metric=name, drivers=" · ".join(f"{NAMES.get(atts[j], atts[j])} {wm[j]:+.2f}" for j in o)))
        tj = int(np.argmax(w))
        col = X[:, tj]
        qs = np.quantile(col, [.25, .5, .75])
        edges = [(-1e9, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], 1e9)]
        for bi, (lo, hi) in enumerate(edges):
            m = (col > lo) & (col <= hi)
            if m.sum() == 0:
                continue
            lab = f"≤{hi:g}" if bi == 0 else (f">{lo:g}" if bi == 3 else f"{lo:g}–{hi:g}")
            tier_rows.append(dict(group=G, attribute=NAMES.get(atts[tj], atts[tj]), tier=lab, players=int(m.sum()), avgScore=r(np.asarray(comp)[m].mean())))

    for G, (qual, mets) in M.items():
        L, R = [], []
        for p in P:
            pid = str(p["rosterId"])
            if GRP.get(p.get("position")) != G or not team_of(pid) or team_of(pid) not in CTX:
                continue
            try:
                if not qual(pid):
                    continue
                vals = [fn(pid) for fn in mets.values()]
            except (KeyError, TypeError):
                continue
            if all(np.isfinite(vals)):
                L.append(p)
                R.append(vals)
        if len(L) < 10:
            continue
        R = np.asarray(R)
        Z = np.column_stack([z(R[:, j]) for j in range(R.shape[1])])
        comp = Z.mean(1)
        ovr = np.array([[f(p.get("playerBestOvr"))] for p in L])
        dev = np.array([1.0 if int(f(p.get("devTrait"))) >= 2 else 0.0 for p in L])
        age = np.array([f(p.get("age")) for p in L])
        tms = [team_of(str(p["rosterId"])) for p in L]
        Xc = np.column_stack([ovr[:, 0], age] + [[CTX[t][k] for t in tms] for k in CTXK[G]])
        cap = np.array([[np.log(max(f(p.get("capHit")), 750000) / 1e6 + 0.1)] for p in L])
        atts = ATT[G].split(",") + ["height", "weight"]
        Xa = np.array([[att(p, a) for a in atts] for p in L])
        lam = max(2.0, len(L) * 0.05)
        _, pO = ridge(ovr, comp, 1)
        wod, _ = ridge(np.column_stack([ovr[:, 0], dev, age]), comp, 1)
        _, pC = ridge(Xc, comp, lam)
        _, pS = ridge(cap, comp, 1)
        summary.append(dict(group=G, players=len(L), metrics=" · ".join(mets.keys()), r2Ovr=r(cv_r2(ovr, comp, 1)),
                            r2Attr=r(cv_r2(Xa, comp, lam)), r2Salary=r(cv_r2(cap, comp, 1)), devBeyondOvr=r(wod[1])))
        eO, eC, eS = comp - pO(ovr), comp - pC(Xc), comp - pS(cap)
        order = np.argsort(comp)
        pct = np.empty(len(comp))
        pct[order] = np.arange(len(comp)) / max(1, len(comp) - 1) * 100
        mk = list(mets.keys())
        for i, p in enumerate(L):
            s = next((st[c][str(p["rosterId"])] for c in ["passing", "rushing", "receiving", "defense"] if str(p["rosterId"]) in st[c] and (
                {"QB": "passing", "HB": "rushing", "WR": "receiving", "TE": "receiving"}.get(G, "defense") == c)), {})
            norm_rows.append(dict(group=G, player=p.get("fullName"), pos=p.get("position"), team=tms[i], nowTeam=abbr(p) or "FA",
                                  age=int(f(p.get("age"))), ovr=int(f(p.get("playerBestOvr"))), dev=DEV[min(3, int(f(p.get("devTrait"))))],
                                  capHitM=r(f(p.get("capHit")) / 1e6), yearsLeft=int(f(p.get("contractYearsLeft"))), games=int(f(s.get("gamesPlayed"))),
                                  metrics=" · ".join(f"{mk[j]}: {R[i, j]:.2f}" for j in range(len(mk))),
                                  normScore=r(comp[i]), pctile=int(round(pct[i])), overOvr=r(eO[i]), overCtx=r(eC[i]), overSalary=r(eS[i])))
        attr_block(G, atts, Xa, comp, [(mk[j], Z[:, j]) for j in range(len(mk))])

    # OL komandų lygiu
    rowsOL = []
    for t in tnames:
        ps = [s for s in reg["passing"] if s.get("team__abbrName") == t]
        hbs = [s for s in reg["rushing"] if s.get("team__abbrName") == t and (byid.get(str(s.get("player__rosterId"))) or {}).get("position") == "HB"]
        if not ps or not hbs or len(CTX[t]["ol"]) < 5:
            continue
        att_ = sum(f(s.get("passTotalAtt")) for s in ps)
        sk = sum(f(s.get("passTotalSacks")) for s in ps)
        ra = sum(f(s.get("rushTotalAtt")) for s in hbs)
        ybc = (sum(f(s.get("rushTotalYds")) for s in hbs) - sum(f(s.get("rushTotalYdsAfterContact")) for s in hbs)) / max(ra, 1)
        qb = byid.get(str(max(ps, key=lambda s: f(s.get("passTotalAtt"))).get("player__rosterId")))
        hb = byid.get(str(max(hbs, key=lambda s: f(s.get("rushTotalAtt"))).get("player__rosterId")))
        if not qb or not hb:
            continue
        rowsOL.append(dict(t=t, sackPct=100 * sk / max(att_ + sk, 1), ybc=ybc, qbA=f(qb.get("awareRating")), hbS=f(hb.get("speedRating")),
                           X=[float(np.mean([att(p, a) for p in CTX[t]["ol"]])) for a in OL_ATT + ["height", "weight"]]))
    olinfo = {}
    if len(rowsOL) >= 15:
        def resid(y, x):
            _, pr = ridge(np.array(x)[:, None], y, 0.5)
            return np.asarray(y) - pr(np.array(x)[:, None])
        skA = z(resid([-x["sackPct"] for x in rowsOL], [x["qbA"] for x in rowsOL]))
        ybA = z(resid([x["ybc"] for x in rowsOL], [x["hbS"] for x in rowsOL]))
        comp = (skA + ybA) / 2
        attr_block("OL", OL_ATT + ["height", "weight"], [x["X"] for x in rowsOL], comp,
                   [("Sack% (mažiau = geriau), atmetus QB Awareness", skA), ("Jardai iki kontakto, atmetus HB greitį", ybA)])
        summary.append(dict(group="OL", players=len(rowsOL), metrics="komandų lygiu: sack% ir jardai iki kontakto", r2Ovr=None, r2Attr=None, r2Salary=None, devBeyondOvr=None))
        olinfo = {x["t"]: dict(sackPct=r(x["sackPct"], 1), ybc=r(x["ybc"])) for x in rowsOL}

    # ---- mano komanda
    SLOTS = {"QB": 1, "HB": 1, "WR": 3, "TE": 1, "LT": 1, "LG": 1, "C": 1, "RG": 1, "RT": 1, "LEDGE": 1, "REDGE": 1, "DT": 2,
             "MIKE": 1, "WILL": 1, "CB": 3, "FS": 1, "SS": 1, "K": 1, "P": 1}
    posrank = []
    for pos, n in SLOTS.items():
        vals = {}
        for t in tnames:
            l = top(t, [pos], n)
            vals[t] = sum(f(p.get("playerBestOvr")) for p in l) / n if l else 0
        srt = sorted(vals.items(), key=lambda kv: -kv[1])
        best = srt[0]
        nz = [v for v in vals.values() if v > 0]
        posrank.append(dict(pos=pos, myOvr=r(vals.get(MYTEAM, 0), 1), rank=[k for k, _ in srt].index(MYTEAM) + 1 if MYTEAM in vals else None,
                            lgAvg=r(np.mean(nz) if nz else 0, 1), best=f"{best[0]} {best[1]:.0f}"))
    nmap = {(x["player"], x["group"]): x for x in norm_rows}
    mine = []
    for p in P:
        if abbr(p) != MYTEAM or p.get("isFreeAgent"):
            continue
        G = GRP.get(p.get("position"), p.get("position"))
        nr = nmap.get((p.get("fullName"), G))
        same = sorted([q for q in P if abbr(q) == MYTEAM and q.get("position") == p.get("position")], key=lambda q: -f(q.get("playerBestOvr")))
        ka = KEYATT.get(G, [])
        row = dict(player=p.get("fullName"), pos=p.get("position"), group=G, depth=[q["rosterId"] for q in same].index(p["rosterId"]) + 1,
                   age=int(f(p.get("age"))), ovr=int(f(p.get("playerBestOvr"))), dev=DEV[min(3, int(f(p.get("devTrait"))))],
                   capHitM=r(f(p.get("capHit")) / 1e6), yearsLeft=int(f(p.get("contractYearsLeft"))),
                   releaseSaveM=r(f(p.get("capReleaseNetSavings")) / 1e6, 1), tradeValue=int(f(p.get("trade_value"))),
                   keyAttr=" · ".join(f"{NAMES.get(a, a)} {int(att(p, a))}" for a in ka),
                   pctile=nr["pctile"] if nr else None, overCtx=nr["overCtx"] if nr else None, overSalary=nr["overSalary"] if nr else None,
                   metrics=nr["metrics"] if nr else "")
        # automatinis įvertinimas
        if nr is None:
            v = "Mažai žaidė" if row["depth"] > 1 else "Be statistikos"
        elif nr["pctile"] >= 85:
            v = "Elitas"
        elif nr["overCtx"] >= 0.4:
            v = "Žaidžia virš OVR"
        elif nr["overCtx"] <= -0.4:
            v = "Žaidžia žemiau OVR"
        elif nr["pctile"] >= 50:
            v = "Virš vidurkio"
        else:
            v = "Žemiau vidurkio"
        if nr and nr["overSalary"] is not None and nr["overSalary"] <= -0.6 and row["capHitM"] and row["capHitM"] >= 8:
            v += " · brangus"
        row["auto"] = v
        mine.append(row)
    mine.sort(key=lambda x: (x["group"], x["pos"], x["depth"]))
    t = tinfo.get(MYTEAM, {})
    me = dict(team=MYTEAM, name=t.get("displayName"), wins=t.get("totalWins"), losses=t.get("totalLosses"), seed=t.get("seed"),
              ptsForRank=t.get("ptsForRank"), ptsAgainstRank=t.get("ptsAgainstRank"), offPassYdsRank=t.get("offPassYdsRank"),
              offRushYdsRank=t.get("offRushYdsRank"), defPassYdsRank=t.get("defPassYdsRank"), defRushYdsRank=t.get("defRushYdsRank"),
              capAvailableM=r(f(t.get("capAvailable")) / 1e6, 1), capSpentM=r(f(t.get("capSpent")) / 1e6, 1),
              sackPct=(olinfo.get(MYTEAM) or {}).get("sackPct"), lgSackPct=r(np.mean([v["sackPct"] for v in olinfo.values()]), 1) if olinfo else None)
    return dict(norm=norm_rows, summary=summary, attr=attr_rows, attrMetric=metric_rows, attrTiers=tier_rows,
                mine=mine, posrank=posrank, me=me)


def main():
    teams, players, games, stats, season = fetch_all()
    print(f"teams {len(teams)} players {len(players)} games {len(games)} stats {[(k, len(v)) for k, v in stats.items()]}")
    if len(players) < 500 or len(teams) < 20:
        sys.exit("Per mažai duomenų – API grąžino netikėtą atsakymą")
    idx, new = archive(teams, players, games, stats, season)
    print("archyvas:", "nauja kopija" if new else "be pokyčių")
    out = analyze(teams, players, stats)
    out["meta"] = dict(updated=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), league=LEAGUE, season=season,
                       archive=[dict(date=e["date"], files=e["files"]) for e in idx][::-1])
    json.dump(out, open(os.path.join(DATA, "site.json"), "w"), ensure_ascii=False, separators=(",", ":"))
    print("site.json:", {k: len(v) for k, v in out.items() if isinstance(v, list)})


if __name__ == "__main__":
    main()
