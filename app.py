"""Perfume Finder - search every shop in links.md for a perfume.

Run:   python app.py
Open:  http://127.0.0.1:5000
"""

from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

import shops

ROOT = Path(__file__).resolve().parent
LINKS_FILE = ROOT / "links.md"

app = Flask(__name__, static_folder=None)


def shop_bases() -> list[str]:
    return shops.read_links(LINKS_FILE)  # re-read each time so edits to links.md apply instantly


@app.get("/")
def index():
    return send_from_directory(ROOT / "static", "index.html")


@app.get("/api/shops")
def list_shops():
    return jsonify([{"shop": b.split("://", 1)[1].removeprefix("www."), "site": b} for b in shop_bases()])


@app.get("/api/search")
def search():
    q = request.args.get("q", "").strip()
    site = request.args.get("site", "")
    if len(q) < 2:
        abort(400, "Query must be at least 2 characters")
    if site not in shop_bases():  # only search shops from links.md
        abort(400, "Unknown shop")
    return jsonify(shops.search_shop(site, q))


if __name__ == "__main__":
    print(f"Shops loaded from {LINKS_FILE.name}: {', '.join(shop_bases()) or 'none'}")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
