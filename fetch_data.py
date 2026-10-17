#!/usr/bin/env python3
"""Pasiima Sleeper duomenis ir sugeneruoja data.json: sudėtis, sezono statistika, waiver ir trade pasiūlymai."""
import datetime
import json
import os
import sys
import urllib.request

API = "https://api.sleeper.app/v1"
STATS_API = "https://api.sleeper.com/stats/nfl"
CAPS = {"QB": 36, "RB": 80, "WR": 100, "TE": 40, "K": 25, "DEF": 32}  # kiek žaidėjų per poziciją rodyti statistikoje
ELIG = {"QB": {"QB", "SUPER_FLEX"},
        "RB": {"RB", "FLEX", "WRRB_FLEX", "SUPER_FLEX"},
        "WR": {"WR", "FLEX", "WRRB_FLEX", "REC_FLEX", "SUPER_FLEX"},
        "TE": {"TE", "FLEX", "REC_FLEX", "SUPER_FLEX"},
        "K": {"K"}, "DEF": {"DEF"}}
OUT = {"Out", "IR", "Sus", "PUP", "NA"}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "fantasy-dashboard"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def ppr_points(s):
    if s.get("pts_ppr") is not None:
        return round(float(s["pts_ppr"]), 1)
    g = lambda k: float(s.get(k, 0) or 0)
    return round(g("pass_yd") * 0.04 + g("pass_td") * 4 - g("pass_int") * 2 + g("rush_yd") * 0.1
                 + g("rush_td") * 6 + g("rec") + g("rec_yd") * 0.1 + g("rec_td") * 6 - g("fum_lost") * 2
                 + (g("pass_2pt") + g("rush_2pt") + g("rec_2pt")) * 2, 1)


def season_stats(season, upto):
    """pid -> {"w": {savaitė: PPR taškai}, "tgt": taikiniai, "att": nešimai}"""
    per = {}
    for wk in range(1, upto + 1):
        try:
            data = get(f"{STATS_API}/{season}/{wk}?season_type=regular")
        except Exception as e:  # viena savaitė nesužlugdo visko
            print(f"{wk} savaitės statistikos gauti nepavyko:", e, file=sys.stderr)
            continue
        items = data.items() if isinstance(data, dict) else ((r.get("player_id"), r) for r in data)
        for pid, row in items:
            st = row.get("stats", row) if isinstance(row, dict) else None
            if not pid or not st:
                continue
            pts = ppr_points(st)
            if not ((st.get("gp") or 0) > 0 or pts != 0):
                continue
            d = per.setdefault(str(pid), {"w": {}, "tgt": 0, "att": 0})
            d["w"][wk] = pts
            d["tgt"] += st.get("rec_tgt", 0) or 0
            d["att"] += st.get("rush_att", 0) or 0
    return per


def build_stat(pid, players, per, upto, adds):
    p = players.get(pid, {})
    name = p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip() or pid
    d = per.get(pid) or {"w": {}, "tgt": 0, "att": 0}
    w = d["w"]
    gp, total = len(w), round(sum(w.values()), 1)
    avg = round(total / gp, 1) if gp else 0.0
    last3 = [w[k] for k in range(max(1, upto - 2), upto + 1) if k in w]
    l3 = round(sum(last3) / len(last3), 1) if last3 else 0.0
    return {"name": name, "pos": p.get("position") or "", "team": p.get("team") or "FA",
            "status": p.get("injury_status"), "gp": gp, "total": total, "avg": avg, "l3": l3,
            "last": w.get(upto), "form": round(0.6 * l3 + 0.4 * avg, 1),
            "tgt": d["tgt"], "att": d["att"], "adds": adds.get(pid, 0)}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    username = os.environ.get("SLEEPER_USERNAME") or cfg["sleeper_username"]

    state = get(f"{API}/state/nfl")
    season, week = str(state["season"]), int(state.get("week") or 0)
    upto = cfg.get("stats_week") or max(week - 1, 1)

    uid = get(f"{API}/user/{username}")["user_id"]
    players = get(f"{API}/players/nfl")
    per = season_stats(season, upto)
    trending = get(f"{API}/players/nfl/trending/add?lookback_hours=24&limit=40")
    adds = {t["player_id"]: t["count"] for t in trending}

    # sezono lygos duomenys
    raw = []
    for lg in get(f"{API}/user/{uid}/leagues/nfl/{season}"):
        lid = lg["league_id"]
        rosters = get(f"{API}/league/{lid}/rosters")
        mine = next((r for r in rosters if r.get("owner_id") == uid), None)
        if mine:
            users = {u["user_id"]: (u.get("metadata") or {}).get("team_name") or u.get("display_name")
                     for u in get(f"{API}/league/{lid}/users")}
            raw.append((lg, rosters, mine, users))

    # kokių žaidėjų statistiką rodyti: geriausi pagal poziciją + mano sudėtys + populiarūs pridėjimai
    ids = set(adds) | {p for _, _, m, _ in raw for p in (m.get("players") or [])}
    for pos, cap in CAPS.items():
        pool = [pid for pid, p in players.items() if p.get("position") == pos and pid in per]
        ids |= set(sorted(pool, key=lambda x: -sum(per[x]["w"].values()))[:cap])
    S = {pid: build_stat(pid, players, per, upto, adds) for pid in ids if players.get(pid, {}).get("position") in CAPS}

    leagues = []
    for lg, rosters, mine, users in raw:
        slots = [s for s in (lg.get("roster_positions") or []) if s != "BN"]
        owner_of, other_starters = {}, {}
        for r in rosters:
            name = users.get(r.get("owner_id")) or f"Komanda {r.get('roster_id')}"
            for p in r.get("players") or []:
                owner_of[p] = name
            other_starters[name] = set(r.get("starters") or [])
        starters = list(mine.get("starters") or [])
        my_ids = set(mine.get("players") or [])
        bench = [p for p in my_ids if p not in starters]

        def weakest(pos):
            vals = []
            for i, pid in enumerate(starters):
                if (slots[i] if i < len(slots) else "") in ELIG.get(pos, ()):
                    s = S.get(pid)
                    dead = (not pid) or pid == "0" or (s and s["status"] in OUT)
                    vals.append((0.0 if dead or not s else s["form"], pid))
            return min(vals) if vals else None

        waivers = []
        for pid, s in S.items():
            if pid in owner_of or s["status"] in OUT:
                continue
            w = weakest(s["pos"])
            if w and s["form"] - w[0] >= 1:
                waivers.append({"id": pid, "gain": round(s["form"] - w[0], 1), "replaces": w[1], "replaces_form": w[0]})
        waivers = sorted(waivers, key=lambda x: -x["gain"])[:8]

        offers = sorted([b for b in bench if b in S and S[b]["status"] not in OUT], key=lambda b: -S[b]["form"])
        trades, per_owner = [], {}
        for pid, owner in owner_of.items():
            s = S.get(pid)
            if pid in my_ids or not s or pid in other_starters.get(owner, set()) or s["status"] in OUT:
                continue
            w = weakest(s["pos"])
            if w and s["form"] - w[0] >= 2:
                offer = [b for b in offers if S[b]["pos"] != s["pos"]][:2] or offers[:2]
                trades.append({"id": pid, "owner": owner, "gain": round(s["form"] - w[0], 1),
                               "replaces": w[1], "offer": offer})
        trades.sort(key=lambda x: -x["gain"])
        picked = []
        for t in trades:
            if per_owner.get(t["owner"], 0) < 2:
                picked.append(t)
                per_owner[t["owner"]] = per_owner.get(t["owner"], 0) + 1
            if len(picked) == 8:
                break

        leagues.append({
            "id": lg["league_id"], "name": lg["name"], "team": users.get(mine.get("owner_id")),
            "roster": [{"id": pid, "slot": slots[i] if i < len(slots) else "?"}
                       for i, pid in enumerate(starters) if pid and pid != "0"],
            "bench": bench, "owners": {pid: o for pid, o in owner_of.items() if pid in S and pid not in my_ids},
            "waivers": waivers, "trades": picked})

    out = {"sample": False, "updated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="minutes"),
           "season": season, "week": week, "stats_week": upto, "user": username, "stats": S, "leagues": leagues}
    path = os.environ.get("DATA_OUT") or os.path.join(here, "data.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Išsaugota: {len(leagues)} lyga(-os), {len(S)} žaidėjų statistikoje, {season} sezonas, {week} savaitė")


if __name__ == "__main__":
    main()
