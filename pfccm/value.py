"""Pozicijų vertės rodikliai ir atributų analizė.

Kiekvienai pozicijai:
  * efektyvumas – vertė per galimybę (QB: per žaidimą, HB: per prisilietimą, WR/TE: per taikinį) arba per rungtynes (gynyba, K, P);
  * mažos imties korekcija – prie žaidėjo pridedama K „vidutinių“ galimybių;
  * naudingumas – (korekcija − pakaitinis lygis) × galimybės per rungtynes × rungtynės = sezono vertė virš pakaitinio;
  * atributų poveikis – svertinis (pagal patikimumą) ridge, bootstrap stabilumas, pakopos, komponentų ryšiai.
NeonSportz snap'ų neduoda, todėl žaidimo laikas = galimybės per rungtynes.
"""
import numpy as np

GROUP = {"QB": "QB", "HB": "HB", "WR": "WR", "TE": "TE", "LEDGE": "EDGE", "REDGE": "EDGE", "DT": "DT",
         "MIKE": "LB", "WILL": "LB", "SAM": "LB", "CB": "CB", "FS": "S", "SS": "S", "K": "K", "P": "P"}
ATT = {
    "QB": "aware,throwAccShort,throwAccMid,throwAccDeep,throwPower,throwUnderPressure,throwOnRun,playAction,breakSack,speed,agility",
    "HB": "speed,accel,agility,changeOfDirection,breakTackle,bCV,jukeMove,spinMove,stiffArm,truck,carry,aware,catch,strength",
    "WR": "speed,accel,agility,changeOfDirection,release,routeRunShort,routeRunMed,routeRunDeep,catch,cIT,specCatch,jump,aware",
    "TE": "speed,release,routeRunShort,routeRunMed,routeRunDeep,catch,cIT,specCatch,aware,strength,breakTackle,jump",
    "EDGE": "speed,accel,agility,strength,powerMoves,finesseMoves,blockShed,pursuit,tackle,playRec,aware,hitPower",
    "DT": "speed,accel,agility,strength,powerMoves,finesseMoves,blockShed,pursuit,tackle,playRec,aware,hitPower",
    "LB": "speed,accel,playRec,zoneCover,manCover,tackle,pursuit,blockShed,hitPower,aware,powerMoves,finesseMoves",
    "CB": "speed,accel,agility,changeOfDirection,manCover,zoneCover,press,playRec,catch,aware,jump,tackle,pursuit",
    "S": "speed,accel,agility,manCover,zoneCover,press,playRec,catch,aware,jump,tackle,pursuit,hitPower",
    "K": "kickPower,kickAcc,aware",
    "P": "kickPower,kickAcc,aware",
}
NAMES = {"aware": "Awareness", "throwAccShort": "Short Acc", "throwAccMid": "Mid Acc", "throwAccDeep": "Deep Acc", "throwPower": "Throw Power",
         "throwUnderPressure": "Under Pressure", "throwOnRun": "On the Run", "playAction": "Play Action", "breakSack": "Break Sack",
         "speed": "Speed", "agility": "Agility", "accel": "Acceleration", "changeOfDirection": "Change of Dir.", "breakTackle": "Break Tackle",
         "bCV": "BC Vision", "jukeMove": "Juke", "spinMove": "Spin", "stiffArm": "Stiff Arm", "truck": "Trucking", "carry": "Carrying",
         "catch": "Catching", "strength": "Strength", "release": "Release", "routeRunShort": "Short Route", "routeRunMed": "Medium Route",
         "routeRunDeep": "Deep Route", "cIT": "Catch in Traffic", "specCatch": "Spectacular Catch", "jump": "Jumping", "powerMoves": "Power Moves",
         "finesseMoves": "Finesse Moves", "blockShed": "Block Shedding", "pursuit": "Pursuit", "tackle": "Tackle", "playRec": "Play Recognition",
         "hitPower": "Hit Power", "zoneCover": "Zone Coverage", "manCover": "Man Coverage", "press": "Press", "height": "Ūgis",
         "weight": "Svoris", "age": "Amžius", "kickPower": "Kick Power", "kickAcc": "Kick Accuracy"}
DEV = ["N", "Star", "SS", "XF"]

# formulės (rodomos puslapyje)
FORMULA = {
    "QB": ("vertė per žaidimą", "žaidimai/rungt.", "(pass jardai + 20×TD − 45×INT − 7×sack + bėg. jardai + 20×bėg. TD − 35×fumble) / (metimai + sack'ai + bėgimai)", 150),
    "HB": ("vertė per prisilietimą", "prisilietimai/rungt.", "(bėg. + gaud. jardai + 20×TD − 35×fumble − 10×drop) / (nešimai + sugavimai)", 100),
    "WR": ("vertė per taikinį", "taikiniai/rungt.", "(jardai + 20×TD) / taikiniai; taikiniai = sugavimai / catch%", 60),
    "TE": ("vertė per taikinį", "taikiniai/rungt.", "(jardai + 20×TD) / taikiniai; taikiniai = sugavimai / catch%", 50),
    "EDGE": ("sutaupyti jardai per rungtynes", "rungtynės", "jardų ekvivalentas, sutaupytas per rungtynes: INT 50, FF 20, FR 20, TD +20, sack 14, PD 7, tackle 3 (pakoreguotas pagal komandos gynybos apimtį)", 4),
    "DT": ("sutaupyti jardai per rungtynes", "rungtynės", "jardų ekvivalentas, sutaupytas per rungtynes: INT 50, FF 20, FR 20, TD +20, sack 14, PD 7, tackle 3 (pakoreguotas pagal komandos gynybos apimtį)", 4),
    "LB": ("sutaupyti jardai per rungtynes", "rungtynės", "jardų ekvivalentas, sutaupytas per rungtynes: INT 50, FF 20, FR 20, TD +20, sack 14, PD 7, tackle 3 (pakoreguotas pagal komandos gynybos apimtį)", 4),
    "CB": ("sutaupyti jardai per rungtynes", "rungtynės", "INT 50, FF 20, FR 20, TD +20, sack 14, PD 7, tackle 2, leistas sugavimas −4 (tackle pakoreguotas pagal komandos apimtį)", 4),
    "S": ("sutaupyti jardai per rungtynes", "rungtynės", "INT 50, FF 20, FR 20, TD +20, sack 14, PD 7, tackle 3; leistas sugavimas −4 (tackle pakoreguotas pagal komandos apimtį)", 4),
    "K": ("taškai per rungtynes virš vidurkio", "rungtynės", "FG pataikymas virš lygos vidurkio × 3 (50+ jardų ×1.5) + XP virš vidurkio", 4),
    "P": ("grynieji jardai per spyrį", "spyriai/rungt.", "grynieji jardai per spyrį + 5 × spyriai į 20 jardų zoną / spyriai", 30),
}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def wcorr(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    vx, vy = np.average((x - mx) ** 2, weights=w), np.average((y - my) ** 2, weights=w)
    return float(np.average((x - mx) * (y - my), weights=w) / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else 0.0


def wridge(X, y, w, lam):
    mu = np.average(X, 0, w)
    sd = np.sqrt(np.average((X - mu) ** 2, 0, w))
    sd[sd == 0] = 1
    Z = (X - mu) / sd
    ym = np.average(y, weights=w)
    W = w[:, None]
    return np.linalg.solve(Z.T @ (Z * W) + lam * np.eye(Z.shape[1]), Z.T @ (w * (y - ym)))


def now_team(p):
    t = p.get("team")
    return (t.get("abbrName") if isinstance(t, dict) else None) or p.get("teamAbbr") or "FA"


def best_rows(rows):
    m = {}
    for s in rows:
        k = str(s.get("player__rosterId"))
        if k not in m or f(s.get("gamesPlayed")) > f(m[k].get("gamesPlayed")):
            m[k] = s
    return m


def build(players, stats, myteam, starters_per_team):
    """stats: {(cat, stage): rows} vieno sezono. Grąžina {group: {...}}"""
    P = {str(p["rosterId"]): p for p in players if not p.get("isRetired")}
    reg = {c: stats.get((c, 1), []) for c in ["passing", "rushing", "receiving", "defense", "kicking", "punting"]}
    S = {c: best_rows(v) for c, v in reg.items()}
    teams = sorted({s.get("team__abbrName") for s in reg["defense"] if s.get("team__abbrName")})
    tm_tkl = {t: sum(f(s.get("defTotalTackles")) for s in reg["defense"] if s.get("team__abbrName") == t) for t in teams}
    lg_tkl = np.mean(list(tm_tkl.values())) if tm_tkl else 1
    out = {}

    def g(s, k):
        return f(s.get(k)) if s else 0.0

    # ---- surinkti žaidėjų eilutes kiekvienai grupei
    raw = {G: [] for G in ATT}
    for pid, p in P.items():
        G = GROUP.get(p.get("position"))
        if not G:
            continue
        pa, ru, rc, de, ki, pu = (S[c].get(pid) for c in ["passing", "rushing", "receiving", "defense", "kicking", "punting"])
        team = next((s.get("team__abbrName") for s in (pa, ru, rc, de, ki, pu) if s), None)
        if not team:
            continue
        r = dict(pid=pid, p=p, team=team)
        if G == "QB":
            if not pa:
                continue
            att, sk = g(pa, "passTotalAtt"), g(pa, "passTotalSacks")
            ra = g(ru, "rushTotalAtt")
            opp = att + sk + ra
            val = (g(pa, "passTotalYds") + 20 * g(pa, "passTotalTDs") - 45 * g(pa, "passTotalInts") - 7 * sk
                   + g(ru, "rushTotalYds") + 20 * g(ru, "rushTotalTDs") - 35 * g(ru, "rushTotalFum"))
            games = g(pa, "gamesPlayed")
            comps = {"Comp%": 100 * g(pa, "passTotalComp") / max(att, 1), "Y/A": g(pa, "passTotalYds") / max(att, 1),
                     "TD%": 100 * g(pa, "passTotalTDs") / max(att, 1), "INT% (−)": -100 * g(pa, "passTotalInts") / max(att, 1),
                     "Sack% (−)": -100 * sk / max(att + sk, 1), "Bėg. jardai/rungt.": g(ru, "rushTotalYds") / max(games, 1)}
            line = f"{int(g(pa, 'passTotalYds'))} jd, {int(g(pa, 'passTotalTDs'))} TD, {int(g(pa, 'passTotalInts'))} INT, {int(sk)} sack, {int(g(ru, 'rushTotalYds'))} bėg. jd"
        elif G == "HB":
            if not ru:
                continue
            ra, cat = g(ru, "rushTotalAtt"), g(rc, "recTotalCatches")
            opp = ra + cat
            val = (g(ru, "rushTotalYds") + g(rc, "recTotalYds") + 20 * (g(ru, "rushTotalTDs") + g(rc, "recTotalTDs"))
                   - 35 * g(ru, "rushTotalFum") - 10 * g(rc, "recTotalDrops"))
            games = max(g(ru, "gamesPlayed"), g(rc, "gamesPlayed"))
            comps = {"Jardai/nešimą": g(ru, "rushTotalYds") / max(ra, 1), "YAC/nešimą": g(ru, "rushTotalYdsAfterContact") / max(ra, 1),
                     "Išvengti tackle/nešimą": g(ru, "rushTotalBrokenTackles") / max(ra, 1), "20+ jd/nešimą": g(ru, "rushTotal20PlusYds") / max(ra, 1),
                     "Fumble/100 (−)": -100 * g(ru, "rushTotalFum") / max(ra, 1), "Gaud. jardai/rungt.": g(rc, "recTotalYds") / max(games, 1)}
            line = f"{int(ra)} neš., {int(g(ru, 'rushTotalYds'))} jd, {int(g(ru, 'rushTotalTDs') + g(rc, 'recTotalTDs'))} TD, {int(cat)} sug., {int(g(ru, 'rushTotalFum'))} fum."
        elif G in ("WR", "TE"):
            if not rc:
                continue
            cat, pct = g(rc, "recTotalCatches"), g(rc, "recAvgCatchPct")
            tgt = cat / (pct / 100) if pct > 0 else cat
            opp = tgt
            val = g(rc, "recTotalYds") + 20 * g(rc, "recTotalTDs")
            games = g(rc, "gamesPlayed")
            comps = {"Catch%": pct, "Jardai/sugavimą": g(rc, "recTotalYds") / max(cat, 1), "YAC/sugavimą": g(rc, "recAvgYacPerCatch"),
                     "Drop/100 taik. (−)": -100 * g(rc, "recTotalDrops") / max(tgt, 1), "TD/100 taik.": 100 * g(rc, "recTotalTDs") / max(tgt, 1),
                     "Taikiniai/rungt.": tgt / max(games, 1)}
            line = f"{int(cat)} sug. iš ~{int(round(tgt))}, {int(g(rc, 'recTotalYds'))} jd, {int(g(rc, 'recTotalTDs'))} TD, {int(g(rc, 'recTotalDrops'))} drop"
        elif G in ("EDGE", "DT", "LB", "CB", "S"):
            if not de:
                continue
            games = g(de, "gamesPlayed")
            adj = lg_tkl / max(tm_tkl.get(team, lg_tkl), 1)
            tk = g(de, "defTotalTackles") * adj
            # jardų ekvivalentas, sutaupytas komandai (ta pati skalė kaip QB formulėje: INT ≈ 45 jd + vidutinis metimas)
            tw = 2.0 if G == "CB" else 3.0
            val = (tw * tk + 14 * g(de, "defTotalSacks") + 50 * g(de, "defTotalInts") + 7 * g(de, "defTotalDeflections") + 20 * g(de, "defTotalForcedFum")
                   + 20 * g(de, "defTotalFumRec") + 20 * g(de, "defTotalTDs") + 30 * g(de, "defTotalSafeties")
                   - (4 * g(de, "defTotalCatchAllowed") if G in ("CB", "S") else 0))
            opp = games
            comps = {"Tackle'ai/rungt.": g(de, "defTotalTackles") / max(games, 1), "Sack'ai/rungt.": g(de, "defTotalSacks") / max(games, 1),
                     "INT+PD/rungt.": (g(de, "defTotalInts") + g(de, "defTotalDeflections")) / max(games, 1),
                     "FF/rungt.": g(de, "defTotalForcedFum") / max(games, 1)}
            if G in ("CB", "S"):
                comps["Leisti sug./rungt. (−)"] = -g(de, "defTotalCatchAllowed") / max(games, 1)
            line = f"{int(g(de, 'defTotalTackles'))} tkl, {g(de, 'defTotalSacks'):g} sack, {int(g(de, 'defTotalInts'))} INT, {int(g(de, 'defTotalDeflections'))} PD, {int(g(de, 'defTotalForcedFum'))} FF" + (
                f", {int(g(de, 'defTotalCatchAllowed'))} leista" if G in ("CB", "S") else "")
        elif G == "K":
            if not ki:
                continue
            games = g(ki, "gamesPlayed")
            opp = games
            r.update(fga=g(ki, "fGTotalAtt"), fgm=g(ki, "fGTotalMade"), f50a=g(ki, "fGTotal50PlusAtt"), f50m=g(ki, "fGTotal50PlusMade"),
                     xpa=g(ki, "xPTotalAtt"), xpm=g(ki, "xPTotalMade"))
            val = 0  # užpildoma žemiau (reikia lygos vidurkių)
            comps = {"FG%": 100 * r["fgm"] / max(r["fga"], 1), "50+ FG%": 100 * r["f50m"] / max(r["f50a"], 1), "XP%": 100 * r["xpm"] / max(r["xpa"], 1),
                     "Touchback%": 100 * g(ki, "kickoffTotalTBs") / max(g(ki, "kickoffTotalAtt"), 1)}
            line = f"FG {int(r['fgm'])}/{int(r['fga'])}, 50+ {int(r['f50m'])}/{int(r['f50a'])}, XP {int(r['xpm'])}/{int(r['xpa'])}"
        elif G == "P":
            if not pu:
                continue
            n = g(pu, "puntTotalAtt")
            games = g(pu, "gamesPlayed")
            opp = n
            val = g(pu, "puntTotalNetYds") + 5 * g(pu, "puntsTotalIn20")
            comps = {"Grynieji jd/spyrį": g(pu, "puntAvgNetYdsPerAtt"), "Bruto jd/spyrį": g(pu, "puntAvgYdsPerAtt"),
                     "Į 20 jd/spyrį": g(pu, "puntsTotalIn20") / max(n, 1)}
            line = f"{int(n)} spyr., grynieji {g(pu, 'puntAvgNetYdsPerAtt'):.1f} jd, {int(g(pu, 'puntsTotalIn20'))} į 20"
        else:
            continue
        if opp <= 0 or games <= 0:
            continue
        r.update(G=G, opp=opp, val=val, games=games, comps=comps, line=line)
        raw[G].append(r)

    # K vertė: virš lygos vidurkio
    if raw["K"]:
        ks = raw["K"]
        fg = sum(r["fgm"] for r in ks) / max(sum(r["fga"] for r in ks), 1)
        f50 = sum(r["f50m"] for r in ks) / max(sum(r["f50a"] for r in ks), 1)
        xp = sum(r["xpm"] for r in ks) / max(sum(r["xpa"] for r in ks), 1)
        for r in ks:
            r["val"] = 3 * ((r["fgm"] - r["f50m"]) - (r["fga"] - r["f50a"]) * fg) + 4.5 * (r["f50m"] - r["f50a"] * f50) + (r["xpm"] - r["xpa"] * xp)

    for G, rows in raw.items():
        if len(rows) < 8:
            continue
        label, oplabel, formula, K = FORMULA[G]
        opp = np.array([r["opp"] for r in rows])
        val = np.array([r["val"] for r in rows])
        games = np.array([r["games"] for r in rows])
        per_game_basis = G in ("EDGE", "DT", "LB", "CB", "S", "K")
        mu = val.sum() / opp.sum()
        shr = (val + K * mu) / (opp + K)
        rel = opp / (opp + K)
        opg = np.ones(len(rows)) if per_game_basis else opp / games
        # pakaitinis: 20 procentilis tarp startuolių (daugiausia galimybių turintys N×komandų)
        n_start = starters_per_team.get(G, 1) * 32
        order = np.argsort(-opp)[:n_start]
        repl = float(np.percentile(shr[order], 20))
        pgv = (shr - repl) * opg
        sv = pgv * games
        ovr = np.array([f(r["p"].get("playerBestOvr")) for r in rows])
        # vertė virš OVR
        Xo = np.column_stack([ovr, np.ones(len(ovr))])
        bo = np.linalg.lstsq(Xo * np.sqrt(rel)[:, None], shr * np.sqrt(rel), rcond=None)[0]
        over = shr - Xo @ bo
        sd_shr = float(np.sqrt(np.average((shr - np.average(shr, weights=rel)) ** 2, weights=rel))) or 1
        # starter / rank
        rank_sv = np.argsort(np.argsort(-sv)) + 1
        prows = []
        for i, r in enumerate(rows):
            p = r["p"]
            prows.append(dict(player=p.get("fullName"), pos=p.get("position"), team=r["team"], nowTeam=now_team(p),
                              age=int(f(p.get("age"))), ovr=int(ovr[i]), dev=DEV[min(3, int(f(p.get("devTrait"))))], capHitM=round(f(p.get("capHit")) / 1e6, 2),
                              games=int(games[i]), opp=round(float(opp[i]), 1), oppPg=round(float(opp[i] / games[i]), 1),
                              eff=round(float(val[i] / opp[i]), 2), effAdj=round(float(shr[i]), 2), perGame=round(float(pgv[i]), 2),
                              season=round(float(sv[i]), 1), rank=int(rank_sv[i]), overOvr=round(float(over[i] / sd_shr), 2),
                              rel=round(float(rel[i]), 2), line=r["line"], mine=now_team(p) == myteam))
        prows.sort(key=lambda x: -x["season"])
        # atributai
        atts = ATT[G].split(",")
        X = np.array([[f(r["p"].get(a + "Rating")) for a in atts] + [f(r["p"].get("height")), f(r["p"].get("weight")), f(r["p"].get("age"))] for r in rows])
        names = atts + ["height", "weight", "age"]
        keep = X.std(0) > 0
        X, names = X[:, keep], [n for n, k in zip(names, keep) if k]
        lam = max(3.0, len(rows) * 0.05)
        beta = wridge(X, shr, rel, lam)
        rng = np.random.default_rng(1)
        cnt = np.zeros(len(names))
        B = 200
        for _ in range(B):
            ix = rng.integers(0, len(rows), len(rows))
            cnt[np.argsort(-wridge(X[ix], shr[ix], rel[ix], lam))[:3]] += 1
        # antras tikslas – vertė per rungtynes (įskaitant žaidimo laiką), aktualu HB/WR/TE/QB
        sd_pg = float(np.sqrt(np.average((pgv - np.average(pgv, weights=rel)) ** 2, weights=rel))) or 1
        beta_pg = wridge(X, pgv, rel, lam)
        attrs = sorted([dict(attribute=NAMES.get(n, n), key=n, corr=round(wcorr(X[:, j], shr, rel), 2), weight=round(float(beta[j] / sd_shr), 3),
                             top3=int(round(100 * cnt[j] / B)), corrPg=round(wcorr(X[:, j], pgv, rel), 2), weightPg=round(float(beta_pg[j] / sd_pg), 3),
                             avg=round(float(np.average(X[:, j], weights=rel)), 1)) for j, n in enumerate(names)],
                       key=lambda a: -a["weight"])
        # pakopos pagal svarbiausią (ne amžius/ūgis/svoris jei įmanoma)
        top = next((a for a in attrs if a["key"] not in ("age", "height", "weight")), attrs[0])
        j = names.index(top["key"])
        col = X[:, j]
        qs = np.unique(np.quantile(col, [.25, .5, .75], method="lower"))
        edges = [-1e9, *qs, 1e9]
        tiers = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (col > lo) & (col <= hi)
            if m.sum() == 0:
                continue
            lab = f"≤{hi:g}" if lo < -1e8 else (f">{lo:g}" if hi > 1e8 else f"{lo + 1:g}–{hi:g}")
            tiers.append(dict(tier=lab, n=int(m.sum()), eff=round(float(np.average(shr[m], weights=rel[m])), 2)))
        # komponentai
        comps = []
        for cname in rows[0]["comps"]:
            y = np.array([r["comps"][cname] for r in rows])
            if y.std() == 0:
                continue
            cs = sorted([(NAMES.get(n, n), wcorr(X[:, k], y, rel)) for k, n in enumerate(names)], key=lambda t: -abs(t[1]))[:3]
            comps.append(dict(metric=cname, drivers=" · ".join(f"{n} {c:+.2f}" for n, c in cs), withEff=round(wcorr(y, shr, rel), 2)))
        out[G] = dict(label=label, oppLabel=oplabel, formula=formula, K=K, mean=round(float(mu), 2), repl=round(repl, 2),
                      n=len(rows), r2ovr=round(wcorr(ovr, shr, rel) ** 2, 2), players=prows, attrs=attrs, tierAttr=top["attribute"],
                      tiers=tiers, comps=comps)
    return out


STARTERS = {"QB": 1, "HB": 1, "WR": 3, "TE": 1, "EDGE": 2, "DT": 2, "LB": 2, "CB": 3, "S": 2, "K": 1, "P": 1}
