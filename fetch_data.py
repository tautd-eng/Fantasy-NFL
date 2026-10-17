#!/usr/bin/env python3
"""Pasiima tavo Sleeper lygų duomenis ir sugeneruoja data.json (be papildomų bibliotekų)."""
import datetime
import json
import os
import sys
import urllib.request

API = "https://api.sleeper.app/v1"
STATS_API = "https://api.sleeper.com/stats/nfl"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "fantasy-dashboard"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def ppr_points(s):
    """Naudoja Sleeper pts_ppr; jei jo nėra, skaičiuoja standartinę PPR sistemą."""
    if s.get("pts_ppr") is not None:
        return round(float(s["pts_ppr"]), 1)
    g = lambda k: float(s.get(k, 0) or 0)
    pts = (g("pass_yd") * 0.04 + g("pass_td") * 4 - g("pass_int") * 2
           + g("rush_yd") * 0.1 + g("rush_td") * 6
           + g("rec") + g("rec_yd") * 0.1 + g("rec_td") * 6
           - g("fum_lost") * 2 + (g("pass_2pt") + g("rush_2pt") + g("rec_2pt")) * 2)
    return round(pts, 1)


def week_points(season, week):
    out = {}
    try:
        data = get(f"{STATS_API}/{season}/{week}?season_type=regular")
    except Exception as e:  # statistika nėra kritinė: puslapis veiks ir be jos
        print("Statistikos gauti nepavyko:", e, file=sys.stderr)
        return out
    items = data.items() if isinstance(data, dict) else ((r.get("player_id"), r) for r in data)
    for pid, row in items:
        st = row.get("stats", row) if isinstance(row, dict) else None
        if pid and st:
            out[str(pid)] = ppr_points(st)
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    username = os.environ.get("SLEEPER_USERNAME") or cfg["sleeper_username"]

    state = get(f"{API}/state/nfl")
    season, week = str(state["season"]), int(state.get("week") or 0)
    stats_week = cfg.get("stats_week") or max(week - 1, 1)

    uid = get(f"{API}/user/{username}")["user_id"]
    players = get(f"{API}/players/nfl")
    pts = week_points(season, stats_week)
    trending = get(f"{API}/players/nfl/trending/add?lookback_hours=24&limit=40")

    def card(pid):
        p = players.get(pid, {})
        name = p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip() or pid
        return {"id": pid, "name": name, "pos": p.get("position") or "",
                "team": p.get("team") or "FA", "status": p.get("injury_status"), "pts": pts.get(pid)}

    leagues = []
    for lg in get(f"{API}/user/{uid}/leagues/nfl/{season}"):
        rosters = get(f"{API}/league/{lg['league_id']}/rosters")
        mine = next((r for r in rosters if r.get("owner_id") == uid), None)
        if not mine:
            continue
        taken = {p for r in rosters for p in (r.get("players") or [])}
        starters = [p for p in (mine.get("starters") or []) if p and p != "0"]
        bench = [p for p in (mine.get("players") or []) if p not in starters]
        waiver = [dict(card(t["player_id"]), adds=t["count"])
                  for t in trending if t["player_id"] not in taken][:10]
        leagues.append({"name": lg["name"], "starters": [card(p) for p in starters],
                        "bench": [card(p) for p in bench], "waiver": waiver})

    out = {"sample": False,
           "updated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="minutes"),
           "season": season, "week": week, "stats_week": stats_week,
           "user": username, "leagues": leagues}
    with open(os.path.join(here, "data.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"Išsaugota: {len(leagues)} lyga(-os), {season} sezonas, {week} savaitė")


if __name__ == "__main__":
    main()
