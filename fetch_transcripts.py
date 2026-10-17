#!/usr/bin/env python3
"""Paima naujausius kanalų video per YouTube RSS, atsisiunčia transkriptus ir išsaugo į transcripts/.
Paleidžiama GitHub Actions. Reikia: pip install youtube-transcript-api"""
import json, os, re, sys, time, urllib.request, xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "transcripts")
IDX = os.path.join(OUT, "index.json")
MAX_AGE_DAYS = 14      # kiek senų video dar imti
KEEP_DAYS = 45         # kiek laikyti failų repo
MAX_TRIES = 3          # kiek kartų bandyti pasikartojančiai nepavykusį video
NS = {"a": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}


def feed(cid):
    url = "https://www.youtube.com/feeds/videos.xml?channel_id=" + cid
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=30).read()


def resolve_id(ch):
    """Kanalo ID iš @handle (patikimiausia); jei nepavyksta, naudoja channel_id iš channels.json."""
    h = ch.get("handle")
    if h:
        try:
            url = "https://www.youtube.com/" + (h if h.startswith("@") else "@" + h)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9", "Cookie": "CONSENT=YES+1"})
            html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
            for pat in (r'<link rel="canonical" href="https://www\.youtube\.com/channel/(UC[\w-]{22})"', r'"channelId":"(UC[\w-]{22})"', r'"externalId":"(UC[\w-]{22})"', r'channel_id=(UC[\w-]{22})'):
                m = re.search(pat, html)
                if m:
                    return m.group(1)
        except Exception as ex:
            print("Nepavyko nustatyti ID pagal handle", h, ex)
    return ch.get("channel_id")


def parse_feed(data):
    root = ET.fromstring(data)
    out = []
    for e in root.findall("a:entry", NS):
        vid = e.findtext("yt:videoId", namespaces=NS)
        title = e.findtext("a:title", namespaces=NS) or ""
        pub = e.findtext("a:published", namespaces=NS) or ""
        if vid:
            out.append({"id": vid, "title": title, "published": pub})
    return out


def get_transcript(vid):
    """Veikia ir su nauja (1.x), ir su sena youtube-transcript-api versija."""
    from youtube_transcript_api import YouTubeTranscriptApi
    langs = ["en", "en-US", "en-GB"]
    try:
        t = YouTubeTranscriptApi().fetch(vid, languages=langs)
        parts = [getattr(s, "text", None) or s["text"] for s in t]
    except AttributeError:
        parts = [s["text"] for s in YouTubeTranscriptApi.get_transcript(vid, languages=langs)]
    text = " ".join(p.replace("\n", " ") for p in parts)
    return re.sub(r"\s+", " ", text).strip()


def load_idx():
    try:
        return json.load(open(IDX, encoding="utf-8"))
    except Exception:
        return {}


def main():
    os.makedirs(OUT, exist_ok=True)
    idx = load_idx()
    now = datetime.now(timezone.utc)
    new = fail = 0
    for ch in json.load(open(os.path.join(ROOT, "channels.json"), encoding="utf-8")):
        try:
            cid = resolve_id(ch)
            if not cid:
                print("Nėra kanalo ID:", ch["name"]); continue
            vids = parse_feed(feed(cid))
        except Exception as ex:
            print("RSS klaida", ch["name"], ex); continue
        for v in vids:
            try:
                pub = datetime.fromisoformat(v["published"].replace("Z", "+00:00"))
            except Exception:
                continue
            if now - pub > timedelta(days=MAX_AGE_DAYS):
                continue
            st = idx.get(v["id"], {})
            if st.get("ok") or st.get("tries", 0) >= MAX_TRIES:
                continue
            fn = "%s_%s.md" % (pub.strftime("%Y-%m-%d"), v["id"])
            try:
                text = get_transcript(v["id"])
                if not text:
                    raise ValueError("tuščias transkriptas")
                with open(os.path.join(OUT, fn), "w", encoding="utf-8") as f:
                    f.write("# %s\n\nKanalas: %s\nData: %s\nNuoroda: https://www.youtube.com/watch?v=%s\n\n%s\n" % (v["title"], ch["name"], v["published"][:10], v["id"], text))
                idx[v["id"]] = {"ok": True, "file": fn, "title": v["title"], "channel": ch["name"], "published": v["published"]}
                new += 1; print("OK", v["title"])
            except Exception as ex:
                tries = st.get("tries", 0) + 1
                idx[v["id"]] = {"ok": False, "tries": tries, "title": v["title"], "channel": ch["name"], "published": v["published"], "error": str(ex)[:200]}
                fail += 1; print("KLAIDA", v["title"], type(ex).__name__, str(ex)[:150])
            time.sleep(2)
    # valymas
    for vid, st in list(idx.items()):
        try:
            if now - datetime.fromisoformat(st["published"].replace("Z", "+00:00")) > timedelta(days=KEEP_DAYS):
                if st.get("file"):
                    try: os.remove(os.path.join(OUT, st["file"]))
                    except OSError: pass
                del idx[vid]
        except Exception:
            pass
    json.dump(idx, open(IDX, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("Nauji: %d, nepavyko: %d" % (new, fail))


if __name__ == "__main__":
    main()
