#!/usr/bin/env python3
"""Sleeper + nflverse tvarkaraštis (spread/total) + FantasyCalc trade vertės -> data.json (tik standartinė biblioteka).

Šis skriptas tik surenka ir paruošia duomenis (projekcijos 3 savaitėms, komandų jardų projekcijos, tvarkaraščio komentarai).
Waiver, SELL/HOLD žymas ir trade idėjas puslapis skaičiuoja pats (index.html), kad galėtum konstruoti savo mainus."""
import csv, datetime, io, json, os, sys, urllib.request

API = "https://api.sleeper.app/v1"
STATS_API = "https://api.sleeper.com/stats/nfl"
PROJ_API = "https://api.sleeper.com/projections/nfl"
GAMES_URLS = ["https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
              "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"]
FC_API = "https://api.fantasycalc.com/values/current"
TEAMFIX = {"LA": "LAR", "WSH": "WAS", "JAC": "JAX", "OAK": "LV", "SD": "LAC", "STL": "LAR", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU"}
CAPS = {"QB": 36, "RB": 80, "WR": 100, "TE": 40, "K": 25, "DEF": 32}
LONG_OUT = {"IR", "Sus", "PUP", "NA"}
WTS = [1, 0.7, 0.49, 0.343, 0.24]
K_SHRINK = 1.0
LAST_REG_WEEK = 18


def norm(t): return TEAMFIX.get(t, t)
def clamp(x, a, b): return max(a, min(b, x))
def num(x):
    try: return float(x)
    except (TypeError, ValueError): return None


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "fantasy-dashboard"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def get(url): return json.loads(fetch(url))


def health(status, i):
    """Sveikatos koeficientas i-tai savaitei nuo dabar (0 = artimiausia)."""
    if status in LONG_OUT: return 0.0
    if status == "Out": return (0.0, 0.4, 0.8)[min(i, 2)]
    if status == "Doubtful": return (0.4, 0.8, 1.0)[min(i, 2)]
    if status == "Questionable": return (0.92, 1.0, 1.0)[min(i, 2)]
    return 1.0


def ppr_points(s):
    if s.get("pts_ppr") is not None: return round(float(s["pts_ppr"]), 1)
    g = lambda k: float(s.get(k, 0) or 0)
    return round(g("pass_yd") * .04 + g("pass_td") * 4 - g("pass_int") * 2 + g("rush_yd") * .1 + g("rush_td") * 6 + g("rec")
                 + g("rec_yd") * .1 + g("rec_td") * 6 - g("fum_lost") * 2 + (g("pass_2pt") + g("rush_2pt") + g("rec_2pt")) * 2, 1)


def rows_of(data):
    items = data.items() if isinstance(data, dict) else ((r.get("player_id"), r) for r in data)
    for pid, row in items:
        if pid and isinstance(row, dict): yield str(pid), row.get("stats", row) or {}


SUM_KEYS = ["rec_tgt", "rec", "rec_yd", "rec_td", "rush_att", "rush_yd", "rush_td", "pass_att", "pass_cmp", "pass_yd", "pass_td", "pass_int", "off_snp", "tm_off_snp"]


def season_stats(season, upto, players):
    """Žaidėjų savaitiniai PPR taškai + komandų savaitiniai rush/pass jardai (suma iš žaidėjų)."""
    per, team_wk = {}, {}
    for wk in range(1, upto + 1):
        try: data = get(f"{STATS_API}/{season}/{wk}?season_type=regular")
        except Exception as e:
            print(f"{wk} sav. statistika nepasiekiama:", e, file=sys.stderr); continue
        for pid, st in rows_of(data):
            pts = ppr_points(st)
            t = norm(players.get(pid, {}).get("team") or "")
            if t and players.get(pid, {}).get("position") in ("QB", "RB", "WR", "TE", "FB"):
                d = team_wk.setdefault((t, wk), {"r": 0.0, "p": 0.0})
                d["r"] += float(st.get("rush_yd", 0) or 0); d["p"] += float(st.get("pass_yd", 0) or 0)
            if not ((st.get("gp") or 0) > 0 or pts != 0): continue
            d = per.setdefault(pid, {"w": {}, "t": {k: 0 for k in SUM_KEYS}})
            d["w"][wk] = pts
            for k in SUM_KEYS: d["t"][k] += st.get(k, 0) or 0
    return per, team_wk


def sleeper_proj(season, week):
    out = {}
    try: data = get(f"{PROJ_API}/{season}/{week}?season_type=regular")
    except Exception as e:
        print("Sleeper projekcijų nėra:", e, file=sys.stderr); return out
    for pid, st in rows_of(data):
        if st.get("pts_ppr") is not None: out[pid] = float(st["pts_ppr"])
    return out


def load_schedule(season):
    sched, txt = {}, None
    for u in GAMES_URLS:
        try: txt = fetch(u); break
        except Exception as e: print("Tvarkaraštis nepasiekiamas:", u, e, file=sys.stderr)
    if not txt: return sched
    for r in csv.DictReader(io.StringIO(txt)):
        if r.get("season") != str(season) or r.get("game_type") != "REG": continue
        wk, h, a = int(r["week"]), norm(r["home_team"]), norm(r["away_team"])
        tot, sp = num(r.get("total_line")), num(r.get("spread_line"))  # spread_line > 0: namų komanda favoritė
        sched.setdefault(h, {})[wk] = {"opp": a, "home": True, "total": tot, "spread": sp}
        sched.setdefault(a, {})[wk] = {"opp": h, "home": False, "total": tot, "spread": -sp if sp is not None else None}
    return sched


def dvp_index(per, players, sched):
    """Varžovo gynybos 'leidžiami PPR taškai pozicijai' / lygos vidurkis."""
    tot = {}
    for pid, d in per.items():
        p = players.get(pid, {}); pos = p.get("position"); t = norm(p.get("team") or "")
        if pos not in ("QB", "RB", "WR", "TE"): continue
        for wk, pts in d["w"].items():
            g = sched.get(t, {}).get(wk)
            if g: tot[(g["opp"], pos, wk)] = tot.get((g["opp"], pos, wk), 0) + pts
    by = {}
    for (opp, pos, _), v in tot.items(): by.setdefault((opp, pos), []).append(v)
    avg = {k: sum(v) / len(v) for k, v in by.items()}
    lg = {}
    for (opp, pos), a in avg.items(): lg.setdefault(pos, []).append(a)
    lgavg = {p: sum(v) / len(v) for p, v in lg.items()}
    return {k: a / lgavg[k[1]] for k, a in avg.items() if lgavg.get(k[1])}


class Ctx:
    """Komandų reitingai (puolimo ir gynybos jardai) ir žaidimo scenarijus iš spread/total."""
    def __init__(self, team_wk, sched):
        self.sched = sched
        off, allowed = {}, {}
        for (t, wk), v in team_wk.items():
            off.setdefault(t, []).append(v)
            g = sched.get(t, {}).get(wk)
            if g: allowed.setdefault(g["opp"], []).append(v)
        avg = lambda L, k: sum(x[k] for x in L) / len(L)
        R = {t: {"ro": avg(L, "r"), "po": avg(L, "p")} for t, L in off.items() if L}
        self.Lr = sum(r["ro"] for r in R.values()) / len(R) if R else 1.0
        self.Lp = sum(r["po"] for r in R.values()) / len(R) if R else 1.0
        for t, r in R.items():
            r["rd"] = avg(allowed[t], "r") if t in allowed else self.Lr
            r["pd"] = avg(allowed[t], "p") if t in allowed else self.Lp
        self.R = R
        imps = [g["total"] / 2 for tt in sched.values() for g in tt.values() if g.get("total")]
        self.imp_avg = sum(imps) / len(imps) if imps else 0.0
        self.N = len(R)
        rank = lambda key, rev: {t: i + 1 for i, (t, _) in enumerate(sorted(R.items(), key=lambda kv: -kv[1][key] if rev else kv[1][key]))}
        self.rk_ro, self.rk_po = rank("ro", True), rank("po", True)   # 1 = geriausias puolimas
        self.rk_rd, self.rk_pd = rank("rd", False), rank("pd", False)  # 1 = stipriausia gynyba (leidžia mažiausiai)

    def tproj(self, t, wk):
        g = self.sched.get(t, {}).get(wk)
        if not g or t not in self.R: return None
        o, r = g["opp"], self.R[t]
        od = self.R.get(o, {"rd": self.Lr, "pd": self.Lp})
        spr = g.get("spread") or 0.0  # > 0: komanda favoritė
        tot = g.get("total")
        imp = (tot / 2 + spr / 2) if tot else None
        envf = clamp(imp / self.imp_avg, .85, 1.2) ** .5 if (imp and self.imp_avg) else 1.0
        pr = r["ro"] * od["rd"] / self.Lr * clamp(1 + .012 * spr, .88, 1.12) * envf
        pp = r["po"] * od["pd"] / self.Lp * clamp(1 - .012 * spr, .88, 1.12) * envf
        return {"opp": o, "h": g["home"], "pr": pr, "pp": pp, "rr": pr / r["ro"] if r["ro"] else 1.0,
                "pq": pp / r["po"] if r["po"] else 1.0, "spr": spr, "tot": tot, "imp": imp}


def mu_for(c, pos, t, wk, dvp):
    x = c.tproj(t, wk)
    if not x: return None, None
    d = dvp.get((x["opp"], pos))
    dp = 1 + .5 * (d - 1) if d else None
    if pos == "RB": yp = 1 + .6 * (x["rr"] - 1)
    elif pos in ("WR", "TE"): yp = 1 + .6 * (x["pq"] - 1)
    elif pos == "QB": yp = 1 + .6 * (.8 * (x["pq"] - 1) + .2 * (x["rr"] - 1))
    elif pos == "K": yp = 1 + .3 * (x["imp"] / c.imp_avg - 1) if x["imp"] and c.imp_avg else 1.0
    elif pos == "DEF":
        oi = (x["tot"] - x["imp"]) if x["tot"] and x["imp"] else None
        yp = clamp(1 + .6 * (c.imp_avg / max(oi, 1) - 1), .85, 1.2) if oi and c.imp_avg else 1.0
    else: yp = 1.0
    return clamp((dp + yp) / 2 if dp is not None else yp, .88, 1.12), x


def cls(mu): return "e" if mu > 1.04 else ("t" if mu < 0.96 else "m")


def sched_note(c, pos, t, rows, ros):
    if not c.N or pos not in ("QB", "RB", "WR", "TE"): return ""
    parts = []
    for r in rows:
        if r["o"] == "BYE": parts.append(f"S{r['w']} BYE"); continue
        rk = (c.rk_rd if pos == "RB" else c.rk_pd).get(r["o"])
        lab = ("run D" if pos == "RB" else "pass D") + (f" #{rk}" if rk else "")
        q = "" if not rk else (", leidžia daug" if rk > c.N * .7 else (", stipri gynyba" if rk <= c.N * .3 else ", vidutinė"))
        parts.append(f"{'vs' if r['h'] else '@'} {r['o']} ({lab}{q})")
    rr = c.R.get(t)
    ctx = ""
    if rr:
        ctx = (f" Komandos run puolimas #{c.rk_ro[t]} ({rr['ro']:.0f} yd/sav.)." if pos == "RB"
               else f" Komandos pass puolimas #{c.rk_po[t]} ({rr['po']:.0f} yd/sav.).")
    rest = "lengvas" if ros > 1.03 else ("sunkus" if ros < 0.97 else "vidutinis")
    return "Kitos 3 sav.: " + "; ".join(parts) + "." + ctx + f" Likusio sezono kalendorius {rest} (indeksas {ros:.2f})."


def build_stat(pid, players, per, upto, nxt, adds, sp, c, dvp, posbase, team_tgt):
    p = players.get(pid, {})
    name = p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip() or pid
    pos, t = p.get("position") or "", norm(p.get("team") or "")
    d = per.get(pid) or {"w": {}, "t": {k: 0 for k in SUM_KEYS}}
    w, T = d["w"], d["t"]
    gp, total = len(w), round(sum(w.values()), 1)
    avg = round(total / gp, 1) if gp else 0.0
    last3 = [w[k] for k in range(max(1, upto - 2), upto + 1) if k in w]
    l3 = round(sum(last3) / len(last3), 1) if last3 else 0.0
    status = p.get("injury_status")
    # bazė: paskutinių 5 savaičių svertinis vidurkis, mažai žaidusiems traukiamas link pozicijos vidurkio (pakoreguoto komandos puolimu)
    recent = sorted(w)[-5:][::-1]
    wts = WTS[:len(recent)]
    rr = c.R.get(t)
    tr_f = 1.0
    if rr and pos == "RB": tr_f = clamp(rr["ro"] / c.Lr, .8, 1.25)
    elif rr and pos in ("WR", "TE", "QB"): tr_f = clamp(rr["po"] / c.Lp, .8, 1.25)
    base = (sum(a * w[k] for a, k in zip(wts, recent)) + K_SHRINK * posbase.get(pos, 0) * tr_f) / (sum(wts) + K_SHRINK) if recent else 0.0
    sl = sp.get(pid)
    sos, fut, model1, mus = [], [], 0.0, []
    for i, wk in enumerate(range(nxt, min(nxt + 4, LAST_REG_WEEK + 1))):
        g = c.sched.get(t, {}).get(wk)
        bye = bool(c.sched) and t in c.sched and g is None
        if bye:
            sos.append({"w": wk, "o": "BYE", "h": True, "d": "b", "mu": 1.0})
            if i < 3: fut.append({"w": wk, "o": "BYE", "h": True, "p": 0.0, "d": "b", "mu": 1.0})
            continue
        mu, _ = mu_for(c, pos, t, wk, dvp) if g else (None, None)
        mu = mu if mu is not None else 1.0
        mus.append(mu)
        sos.append({"w": wk, "o": g["opp"] if g else "", "h": g["home"] if g else True, "d": cls(mu), "mu": round(mu, 2)})
        if i < 3:
            m = base * mu * health(status, i)
            if i == 0:
                model1 = round(m, 1)
                m = 0.0 if health(status, i) == 0 else (0.5 * m + 0.5 * sl if sl is not None else m)
            fut.append({"w": wk, "o": g["opp"] if g else "", "h": g["home"] if g else True, "p": round(m, 1), "d": cls(mu), "mu": round(mu, 2)})
    ros_all = []
    for wk in range(nxt, LAST_REG_WEEK):
        g = c.sched.get(t, {}).get(wk)
        if g:
            mu, _ = mu_for(c, pos, t, wk, dvp)
            if mu is not None: ros_all.append(mu)
    ros = sum(ros_all) / len(ros_all) if ros_all else 1.0
    sosx = sum(mus) / len(mus) if mus else 1.0
    proj = fut[0]["p"] if fut else 0.0
    first = fut[0] if fut else None
    opp = "" if not first else ("BYE" if first["o"] == "BYE" else (("vs " if first["h"] else "@ ") + first["o"] if first["o"] else ""))
    tt = team_tgt.get(t, 0)
    return {"name": name, "pos": pos, "team": t or "FA", "status": status, "gp": gp, "total": total, "avg": avg, "l3": l3,
            "last": w.get(upto), "weeks": [w.get(k, 0) for k in range(max(1, upto - 4), upto + 1)],
            "proj": proj, "model": model1, "sl": sl, "p3": round(sum(x["p"] for x in fut), 1), "fut": fut,
            "mu": first["mu"] if first else 1.0, "opp": opp, "sos": sos, "sosx": round(sosx, 2), "ros": round(ros, 2),
            "note3": sched_note(c, pos, t, fut, ros),
            "snap": round(T["off_snp"] / T["tm_off_snp"] * 100) if T["tm_off_snp"] else None,
            "tgt": T["rec_tgt"], "tshare": round(T["rec_tgt"] / tt * 100) if tt and T["rec_tgt"] else None,
            "rec": T["rec"], "recyd": T["rec_yd"], "rectd": T["rec_td"], "ratt": T["rush_att"], "ryd": T["rush_yd"], "rtd": T["rush_td"],
            "patt": T["pass_att"], "cmp": T["pass_cmp"], "pyd": T["pass_yd"], "ptd": T["pass_td"], "int": T["pass_int"],
            "adds": adds.get(pid, 0)}


def nfl_team_table(c, nxt):
    out = {}
    for t, r in c.R.items():
        x = c.tproj(t, nxt)
        out[t] = {"ro": round(r["ro"]), "po": round(r["po"]), "rd": round(r["rd"]), "pd": round(r["pd"]),
                  "rkro": c.rk_ro[t], "rkpo": c.rk_po[t], "rkrd": c.rk_rd[t], "rkpd": c.rk_pd[t],
                  "next": None if not x else {"opp": x["opp"], "h": x["h"], "pr": round(x["pr"]), "pp": round(x["pp"]),
                                              "spr": round(x["spr"], 1), "tot": x["tot"], "imp": round(x["imp"], 1) if x["imp"] else None}}
    return out


def league_format(lg):
    ss, rp = lg.get("scoring_settings") or {}, lg.get("roster_positions") or []
    rec = ss.get("rec", 0) or 0
    qbs = 2 if ("SUPER_FLEX" in rp or rp.count("QB") >= 2) else 1
    teams = int(lg.get("total_rosters") or 12)
    ppr = min((0, 0.5, 1), key=lambda x: abs(x - rec))
    notes = []
    if rec not in (0, 0.5, 1): notes.append("Priėmimo taškai nestandartiniai, vertės apytikslės.")
    if rec != 1: notes.append("Taškai skaičiuojami PPR sistema, o tavo lyga turi kitus priėmimo taškus.")
    if (ss.get("pass_td") or 4) != 4: notes.append("QB TD ne 4 taškai: QB vertės gali būti netikslios.")
    if (ss.get("bonus_rec_te") or 0) > 0: notes.append("Lygoje TE premija, kurios trade šaltinis neįvertina.")
    label = {0: "Standard", 0.5: "Half PPR", 1: "PPR"}[ppr] + f" · {'Superflex/2QB' if qbs == 2 else '1QB'} · {teams} komandų"
    return {"label": label, "qbs": qbs, "teams": teams, "ppr": ppr, "notes": notes}


def trade_values(fmt):
    try:
        data = get(f"{FC_API}?isDynasty=false&numQbs={fmt['qbs']}&numTeams={fmt['teams']}&ppr={fmt['ppr']}")
        vals = {str(i["player"]["sleeperId"]): int(i["value"]) for i in data if i.get("player", {}).get("sleeperId")}
        if vals: return vals, "FantasyCalc"
    except Exception as e: print("FantasyCalc nepasiekiamas:", e, file=sys.stderr)
    return {}, "savo modelis (projekcija), nes trade šaltinis nepasiekiamas"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "config.json"), encoding="utf-8") as f: cfg = json.load(f)
    username = os.environ.get("SLEEPER_USERNAME") or cfg["sleeper_username"]
    state = get(f"{API}/state/nfl")
    season, week = str(state["season"]), int(state.get("week") or 1)
    upto, nxt = cfg.get("stats_week") or max(week - 1, 1), week
    uid = get(f"{API}/user/{username}")["user_id"]
    players = get(f"{API}/players/nfl")
    per, team_wk = season_stats(season, upto, players)
    sp, sched = sleeper_proj(season, nxt), load_schedule(season)
    c = Ctx(team_wk, sched)
    dvp = dvp_index(per, players, sched) if sched else {}
    adds = {t["player_id"]: t["count"] for t in get(f"{API}/players/nfl/trending/add?lookback_hours=24&limit=40")}

    raw = []
    for lg in get(f"{API}/user/{uid}/leagues/nfl/{season}"):
        rosters = get(f"{API}/league/{lg['league_id']}/rosters")
        mine = next((r for r in rosters if r.get("owner_id") == uid), None)
        if mine:
            users = {u["user_id"]: (u.get("metadata") or {}).get("team_name") or u.get("display_name") for u in get(f"{API}/league/{lg['league_id']}/users")}
            raw.append((lg, rosters, mine, users))

    ids = set(adds) | {p for _, rs, _, _ in raw for r in rs for p in (r.get("players") or [])}
    for pos, cap in CAPS.items():
        pool = [pid for pid, p in players.items() if p.get("position") == pos and pid in per]
        ids |= set(sorted(pool, key=lambda x: -sum(per[x]["w"].values()))[:cap])
    ids = {i for i in ids if players.get(i, {}).get("position") in CAPS}
    team_tgt = {}
    for pid, d in per.items():
        t = norm(players.get(pid, {}).get("team") or ""); team_tgt[t] = team_tgt.get(t, 0) + d["t"]["rec_tgt"]
    pb = {}
    for pos in CAPS:
        a = [sum(d["w"].values()) / len(d["w"]) for pid, d in per.items() if players.get(pid, {}).get("position") == pos and len(d["w"]) >= 3]
        pb[pos] = (sum(a) / len(a)) if a else 0.0
    S = {pid: build_stat(pid, players, per, upto, nxt, adds, sp, c, dvp, pb, team_tgt) for pid in ids}

    leagues = []
    for lg, rosters, mine, users in raw:
        fmt = league_format(lg)
        fcv, source = trade_values(fmt)
        val = lambda pid: fcv.get(pid, int(S[pid]["avg"] * 60)) if fcv else int(S[pid]["avg"] * 250 + S[pid]["p3"] * 50)
        slots = [s for s in (lg.get("roster_positions") or []) if s != "BN"]
        teams = []
        for r in rosters:
            name = users.get(r.get("owner_id")) or f"Komanda {r.get('roster_id')}"
            starters = [p for p in (r.get("starters") or []) if p and p != "0" and p in S]
            teams.append({"name": name, "mine": r is mine, "players": [p for p in (r.get("players") or []) if p in S], "starters": starters})
        me = next(t for t in teams if t["mine"])
        order = list(mine.get("starters") or [])
        leagues.append({"id": lg["league_id"], "name": lg["name"], "team": me["name"], "slots": slots,
                        "roster": [{"id": pid, "slot": slots[i] if i < len(slots) else "?"} for i, pid in enumerate(order) if pid and pid != "0" and pid in S],
                        "bench": [p for p in me["players"] if p not in me["starters"]],
                        "teams": teams, "values": {p: val(p) for p in S},
                        "format": {"label": fmt["label"], "source": source, "notes": fmt["notes"]}})

    out = {"sample": False, "updated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="minutes"),
           "season": season, "week": nxt, "stats_week": upto, "user": username, "stats": S, "leagues": leagues,
           "nfl_teams": nfl_team_table(c, nxt), "sources": {"schedule": bool(sched), "sleeper_projections": bool(sp)}}
    with open(os.environ.get("DATA_OUT") or os.path.join(here, "data.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"OK: {len(leagues)} lygos, {len(S)} žaidėjų, komandų: {len(c.R)}, tvarkaraštis={bool(sched)}, Sleeper proj.={bool(sp)}")


if __name__ == "__main__":
    main()
