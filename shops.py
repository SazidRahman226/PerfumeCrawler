"""Shop adapters: search one shop for a perfume and return normalized results.

Each shop listed in links.md is auto-detected the first time it is searched:

* WooCommerce  - uses the public Store API (/wp-json/wc/store/v1/products?search=...)
                 and batch-fetches size variations in a single extra request.
* Softs.ai     - Next.js storefront (e.g. onedropbd.com). Its search API rejects
                 direct requests and robots.txt disallows search URLs, so we match the
                 query against the product slugs in sitemap.xml, then read the
                 matching product pages (allowed by robots.txt). Every size's
                 price and availability is embedded in each product page.

Normalized result shape (dicts so they JSON-serialize directly):
    {
      "name", "url", "image", "available": bool,
      "price": float|None, "regular_price": float|None, "currency": "BDT",
      "stock_text": str|None,
      "variants": [{"name", "price", "regular_price", "available", "stock_text"}],
      "score": float   # how well the name matches the query (0-1)
    }
"""

from __future__ import annotations

import html
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import requests

TIMEOUT = 25
USER_AGENT = "PerfumeCrawler/1.0 (personal price checker)"
MAX_PAGE_FETCHES = 10   # Softs.ai: max product pages read per search
CACHE_TTL = 300         # seconds to reuse a search result
SITEMAP_TTL = 1800      # seconds to reuse a sitemap

URL_RE = re.compile(r"https?://[^\s<>\")\]]+")

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json, text/html;q=0.9"})


# --------------------------------------------------------------------------- utils

def read_links(path: Path) -> list[str]:
    """Return unique shop base URLs (scheme://host) from a markdown file."""
    text = re.sub(r"<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.DOTALL)
    bases: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith(("#", ">")):
            continue
        for u in URL_RE.findall(line):
            p = urlparse(u.rstrip(".,;"))
            if not re.fullmatch(r"[a-z0-9-]+(\.[a-z0-9-]+)+", p.netloc.lower()):
                continue  # skips placeholders like https://...
            base = f"{p.scheme}://{p.netloc}"
            if base not in bases:
                bases.append(base)
    return bases


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", html.unescape(text).lower())


# Filler words people type that shops often leave out of product names.
STOPWORDS = {"the", "a", "an", "of", "by", "for", "and", "&", "de", "la", "le", "les", "du", "di", "pour"}


def key_tokens(query: str) -> list[str]:
    """Query words that matter for matching (filler words dropped, unless nothing else is left)."""
    t = tokens(query)
    return [w for w in t if w not in STOPWORDS] or t


def search_variants(query: str) -> list[str]:
    """Search strings to send to a shop, most specific first.

    Shop search engines usually require every word to appear, so an extra word such as
    "the" hides the product. We also try the query without filler words, then the most
    distinctive single words, and rank everything locally with match_score.
    """
    full = " ".join(tokens(query))
    core = key_tokens(query)
    out = [full, " ".join(core)]
    if len(core) > 1:
        out.append(" ".join(core[:2]))                         # usually brand + first name word
        out += sorted((w for w in core if len(w) >= 4), key=len, reverse=True)[:2]
    seen, uniq = set(), []
    for q in out:
        if q and q not in seen:
            seen.add(q); uniq.append(q)
    return uniq


def match_score(query: str, name: str) -> float:
    q = key_tokens(query)
    if not q:
        return 0.0
    name_tokens = tokens(name)
    joined = "".join(name_tokens)
    hits = sum(1 for t in q if t in name_tokens or (len(t) > 3 and t in joined))
    return hits / len(q)


def keep(query: str, score: float) -> bool:
    # single word: must match; multi-word: at least half the words must match
    return score >= (1.0 if len(tokens(query)) == 1 else 0.5)


def clean_html(s) -> str | None:
    if not s:
        return None
    s = re.sub(r"<[^>]+>", " ", str(s))
    s = re.sub(r"\s+", " ", html.unescape(s)).strip()
    return s or None


class _TTLCache:
    def __init__(self):
        self._data: dict = {}
        self._lock = threading.Lock()

    def get(self, key, ttl):
        with self._lock:
            hit = self._data.get(key)
            return hit[1] if hit and time.time() - hit[0] < ttl else None

    def set(self, key, value):
        with self._lock:
            self._data[key] = (time.time(), value)


_cache = _TTLCache()


# --------------------------------------------------------------------------- WooCommerce

class WooCommerceShop:
    kind = "WooCommerce"

    def __init__(self, base: str):
        self.base = base
        self.api = f"{base}/wp-json/wc/store/v1/products"

    @staticmethod
    def _money(prices: dict, key: str) -> float | None:
        raw = prices.get(key)
        if raw in (None, "", "0"):
            return None
        return int(raw) / (10 ** int(prices.get("currency_minor_unit", 0)))

    @staticmethod
    def _variant_name(attrs: list[dict], v: dict) -> str:
        """attrs come from the parent's variations list, e.g. [{"name": "ML", "value": "100"}]."""
        parts = []
        for a in attrs or []:
            label, value = a.get("name", ""), str(a.get("value", ""))
            parts.append(f"{value} {label}".strip() if value.replace(".", "").isdigit() else value)
        if parts:
            return " / ".join(parts)
        return re.sub(r"[^:,]+:\s*", "", v.get("variation") or v.get("name") or "")

    def _search_api(self, q: str) -> list[dict]:
        r = _session.get(self.api, params={"search": q, "per_page": 30}, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()

    def search(self, query: str) -> list[dict]:
        # Try progressively looser searches; stop once a search finds a full match.
        items, seen = [], set()
        for i, q in enumerate(search_variants(query)):
            try:
                batch = self._search_api(q)
            except requests.RequestException:
                if i == 0:
                    raise
                continue
            for p in batch:
                if p.get("id") not in seen:
                    seen.add(p.get("id")); items.append(p)
            if any(match_score(query, html.unescape(p.get("name", ""))) == 1.0 for p in batch):
                break

        results = []
        for p in items:
            name = html.unescape(p.get("name", ""))
            score = match_score(query, name)
            if not keep(query, score):
                continue
            prices = p.get("prices", {})
            results.append({
                "_variations": [(v["id"], v.get("attributes")) for v in p.get("variations", [])],
                "name": name,
                "url": p.get("permalink"),
                "image": (p.get("images") or [{}])[0].get("thumbnail"),
                "available": bool(p.get("is_in_stock")) and bool(p.get("is_purchasable", True)),
                "price": self._money(prices, "price"),
                "regular_price": self._money(prices, "regular_price"),
                "currency": prices.get("currency_code", "BDT"),
                "stock_text": clean_html((p.get("stock_availability") or {}).get("text")),
                "variants": [],
                "score": score,
            })

        # Fetch every size of every variable product in one request.
        all_ids = [vid for res in results for vid, _ in res["_variations"]]
        if all_ids:
            variations = {}
            for i in range(0, len(all_ids), 100):
                chunk = all_ids[i:i + 100]
                vr = _session.get(self.api, params={"include": ",".join(map(str, chunk)),
                                                    "type": "variation", "per_page": 100}, timeout=TIMEOUT)
                vr.raise_for_status()
                variations.update({v["id"]: v for v in vr.json()})
            for res in results:
                for vid, attrs in res["_variations"]:
                    v = variations.get(vid)
                    if not v:
                        continue
                    vp = v.get("prices", {})
                    res["variants"].append({
                        "name": self._variant_name(attrs, v),
                        "price": self._money(vp, "price"),
                        "regular_price": self._money(vp, "regular_price"),
                        "available": bool(v.get("is_in_stock")) and bool(v.get("is_purchasable", True)),
                        "stock_text": clean_html((v.get("stock_availability") or {}).get("text")),
                    })
                _finalize_variants(res)

        for res in results:
            res.pop("_variations")
        return results


# --------------------------------------------------------------------------- Softs.ai (Next.js)

NEXT_PUSH_RE = re.compile(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)</script>', re.S)


class SoftsShop:
    kind = "Softs.ai (Next.js)"

    def __init__(self, base: str):
        self.base = base
        self.image_base = "https://d23qvyoc2z3h66.cloudfront.net/"

    def _product_urls(self) -> list[str]:
        key = ("sitemap", self.base)
        urls = _cache.get(key, SITEMAP_TTL)
        if urls is None:
            r = _session.get(f"{self.base}/sitemap.xml", timeout=TIMEOUT)
            r.raise_for_status()
            urls = re.findall(r"<loc>\s*([^<]*/product/[^<\s]+)\s*</loc>", r.text)
            _cache.set(key, urls)
        return urls

    def _product_details(self, url: str) -> dict | None:
        r = _session.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        payload = "".join(json.loads(f'"{chunk}"') for chunk in NEXT_PUSH_RE.findall(r.text))
        i = payload.find('"productDetails":')
        if i < 0:
            return None
        obj, _ = json.JSONDecoder().raw_decode(payload, i + len('"productDetails":'))
        return obj.get("result")

    @staticmethod
    def _prices(d: dict) -> tuple[float | None, float | None]:
        price, discount = d.get("price"), d.get("discountPrice")
        if discount and price and discount < price:
            return float(discount), float(price)
        return (float(price) if price else None), None

    def search(self, query: str) -> list[dict]:
        candidates = []
        for url in self._product_urls():
            slug = url.rstrip("/").rsplit("/", 1)[-1]
            score = match_score(query, slug.replace("-", " "))
            if keep(query, score):
                candidates.append((score, len(slug), url))
        candidates.sort(key=lambda c: (-c[0], c[1]))
        urls = [u for _, _, u in candidates[:MAX_PAGE_FETCHES]]

        with ThreadPoolExecutor(max_workers=4) as pool:
            details = list(pool.map(self._product_details, urls))

        results = []
        for url, d in zip(urls, details):
            if not d:
                continue
            price, regular = self._prices(d)
            res = {
                "name": d.get("name"),
                "url": url,
                "image": self.image_base + d["mainImageUrl"] if d.get("mainImageUrl") else None,
                "available": bool(d.get("isAvailable")),
                "price": price,
                "regular_price": regular,
                "currency": "BDT",
                "stock_text": None,
                "variants": [],
                "score": match_score(query, d.get("name", "")),
            }
            if d.get("hasVariants"):
                # The shop shows a product as "Out of stock" when the product-level flag is off,
                # even if its individual sizes are still flagged available, so both must be true.
                for v in d.get("variants") or []:
                    vp, vr = self._prices(v)
                    res["variants"].append({"name": v.get("name"), "price": vp, "regular_price": vr,
                                            "available": res["available"] and bool(v.get("isAvailable")),
                                            "stock_text": None})
                _finalize_variants(res)
            results.append(res)
        return results


# --------------------------------------------------------------------------- shared

def _finalize_variants(res: dict) -> None:
    """Derive product-level price/availability from its sizes."""
    vs = res["variants"]
    if not vs:
        return
    res["available"] = any(v["available"] for v in vs)
    in_stock = [v["price"] for v in vs if v["available"] and v["price"] is not None]
    priced = in_stock or [v["price"] for v in vs if v["price"] is not None]
    if priced:
        res["price"] = min(priced)
        res["regular_price"] = None


def detect(base: str):
    """Figure out which adapter a shop needs (cached per process)."""
    key = ("detect", base)
    hit = _cache.get(key, 24 * 3600)
    if hit:
        return hit
    shop = None
    try:
        r = _session.get(f"{base}/wp-json/wc/store/v1/products", params={"per_page": 1}, timeout=TIMEOUT)
        if r.ok and isinstance(r.json(), list):
            shop = WooCommerceShop(base)
    except (requests.RequestException, ValueError):
        pass
    if shop is None:
        try:
            r = _session.get(base, timeout=TIMEOUT)
            if "__next_f" in r.text and "softs.ai" in r.text.lower():
                shop = SoftsShop(base)
        except requests.RequestException:
            pass
    if shop is None:
        raise RuntimeError("Unsupported shop platform (not WooCommerce or Softs.ai)")
    _cache.set(key, shop)
    return shop


def search_shop(base: str, query: str) -> dict:
    """Search a single shop. Never raises; errors are reported in the result."""
    query = query.strip()
    key = ("search", base, query.lower())
    cached = _cache.get(key, CACHE_TTL)
    if cached:
        return cached

    out = {"shop": urlparse(base).netloc.removeprefix("www."), "site": base, "query": query,
           "platform": None, "results": [], "error": None}
    started = time.time()
    try:
        shop = detect(base)
        out["platform"] = shop.kind
        results = shop.search(query)
        if any(r["score"] == 1.0 for r in results):  # exact matches exist: hide partial ones
            results = [r for r in results if r["score"] == 1.0]
        for r in [*results, *(v for r in results for v in r["variants"])]:
            if r["regular_price"] is not None and (r["price"] is None or r["regular_price"] <= r["price"]):
                r["regular_price"] = None  # not actually discounted
        results.sort(key=lambda r: (not r["available"], -r["score"], r["price"] or 1e12))
        out["results"] = results
    except requests.HTTPError as e:
        out["error"] = f"Shop returned HTTP {e.response.status_code}"
    except requests.RequestException as e:
        out["error"] = f"Could not reach shop ({type(e).__name__})"
    except Exception as e:  # parsing changes on the shop's side, etc.
        out["error"] = f"{type(e).__name__}: {e}"
    out["seconds"] = round(time.time() - started, 1)
    if not out["error"]:
        _cache.set(key, out)
    return out
