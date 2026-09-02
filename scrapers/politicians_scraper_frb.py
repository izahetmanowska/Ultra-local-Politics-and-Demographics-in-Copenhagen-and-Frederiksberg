"""
Insta CSV Scraper
-----------------
Reads a CSV named "parties_scraper_frb.csv" with a column "Instagram Tag" (handles without '@').
Fetches each profile's follower count and number of posts using Playwright, then writes
those values into columns "Follower Count" and "Number of Posts".

- Reuses login via ig_auth_state.json after first manual login.
- Preserves all other columns and row order.
- By default writes to parties_scraper_frb_scraped.csv to avoid accidental overwrites.

Usage:
  python insta_playwright_csv.py               # reads parties_scraper_frb.csv, writes parties_scraper_frb_scraped.csv
  python insta_playwright_csv.py --in my.csv --out out.csv --headless

Notes:
- First run: a visible browser opens so you can log in once; afterwards you can use --headless.
- If you're starting from a .numbers file, export to CSV first and name it parties_scraper_frb.csv.
"""

import csv
import re
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

AUTH_STATE = "ig_auth_state.json"

# ------------------------- helpers -------------------------

def parse_count(text: str) -> Optional[int]:
    if not text:
        return None
    s = text.strip().lower().replace(",", "").replace("\u202f", "")
    m = re.match(r"^([\d\.]+)\s*([km]?)", s)
    if m:
        num_str, suffix = m.groups()
        try:
            num = float(num_str)
        except Exception:
            return None
        if suffix == "k":
            num *= 1_000
        if suffix == "m":
            num *= 1_000_000
        return int(num)
    d = re.search(r"[\d,\.]+", text)
    if d:
        try:
            return int(float(d.group(0).replace(",", "")))
        except Exception:
            return None
    return None


def try_selectors(page) -> Tuple[Optional[int], Optional[int]]:
    if page.is_closed():
        return None, None
    posts = followers = None

    # Option 1: header ul > li with readable text
    try:
        lis = page.query_selector_all("header section ul li")
        if lis:
            for li in lis[:10]:
                try:
                    txt = li.inner_text().strip()
                except Exception:
                    txt = (li.get_attribute("title") or "").strip()
                m = re.search(r"([\d\.,\skmKM]+)\s+(posts?|followers?|following)", txt.lower())
                if m:
                    num_text, label = m.groups()
                    count = parse_count(num_text)
                    if count is not None:
                        if label.startswith("post"):
                            posts = count
                        elif label.startswith("follower"):
                            followers = count
            if posts is not None or followers is not None:
                return posts, followers
    except Exception:
        pass

    # Option 2: span[title] counts
    try:
        spans = page.query_selector_all("header section ul li span[title]")
        if spans and len(spans) >= 2:
            posts = parse_count(spans[0].get_attribute("title") or "")
            followers = parse_count(spans[1].get_attribute("title") or "")
            if posts is not None or followers is not None:
                return posts, followers
    except Exception:
        pass

    # Option 3: meta description
    try:
        metas = page.query_selector_all('meta[name="description"]')
        if metas:
            content = (metas[0].get_attribute("content") or "").lower()
            post_m = re.search(r"([\d\.,\skmKM]+)\s+posts?", content)
            foll_m = re.search(r"([\d\.,\skmKM]+)\s+followers?", content)
            posts = parse_count(post_m.group(1)) if post_m else None
            followers = parse_count(foll_m.group(1)) if foll_m else None
            if posts is not None or followers is not None:
                return posts, followers
    except Exception:
        pass

    # Fallback: body text search
    try:
        body = page.inner_text("body")[:20000].lower()
        post_m = re.search(r"([\d\.,\skmKM]+)\s+posts?", body)
        foll_m = re.search(r"([\d\.,\skmKM]+)\s+followers?", body)
        posts = parse_count(post_m.group(1)) if post_m else None
        followers = parse_count(foll_m.group(1)) if foll_m else None
        if posts is not None or followers is not None:
            return posts, followers
    except Exception:
        pass

    return None, None


def ensure_logged_in(context, headless: bool):
    if Path(AUTH_STATE).exists():
        return
    # Show a visible window for the first login
    if headless:
        print("[info] No saved login. First run will open a visible window to log in…")
    page = context.new_page()
    page.goto("https://www.instagram.com/accounts/login/", timeout=60000)
    print(
        "\n=== LOGIN REQUIRED ===\n"
        "1) Log in to Instagram in the opened browser.\n"
        "2) When you're at the feed or a profile, return here.\n"
        "3) Press Enter to continue.\n"
    )
    input("Press Enter here when you are logged in… ")
    context.storage_state(path=AUTH_STATE)
    try:
        page.close()
    except Exception:
        pass


def fetch_one(page, handle: str) -> Tuple[Optional[int], Optional[int]]:
    username = handle.strip().lstrip("@")
    url = f"https://www.instagram.com/{username}/"
    try:
        page.goto(url, timeout=30000, wait_until="networkidle")
    except PlaywrightTimeoutError:
        try:
            page.goto(url, timeout=60000, wait_until="load")
        except Exception:
            return None, None
    except Exception:
        return None, None
    time.sleep(1.2)
    return try_selectors(page)


# ------------------------- CSV I/O -------------------------

def normalize_header(name: str) -> str:
    return (name or "").strip()


def main():
    ap = argparse.ArgumentParser(description="Update parties_scraper_frb.csv with Instagram follower/post counts.")
    ap.add_argument(
    "--in",
    dest="in_path",
    type=Path,
    default=Path("raw-data/social-media/politicians_info_frb.csv"),
    help="Input CSV (default: raw-data/social media/politicians_info_frb.csv)"
)

    ap.add_argument(
    "--out",
    dest="out_path",
    type=Path,
    default=Path("raw-data/social-media/politicians_info_frb_scraped.csv"),
    help="Output CSV (default: raw-data/social media/politicians_info_frb_scraped.csv)"
)

    ap.add_argument("--headless", action="store_true", help="Run browser headless (after first login)")
    args = ap.parse_args()

    in_csv: Path = args.in_path
    out_csv: Path = args.out_path

    if not in_csv.exists():
        print(f"Input CSV not found: {in_csv}")
        sys.exit(1)

    # Read all rows first
    with in_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=';')
        fieldnames = [normalize_header(h) for h in reader.fieldnames or []]
        if not fieldnames:
            print("Input columns:", fieldnames)
            sys.exit(1)

        # Try to find the Instagram column exactly, else fallback case-insensitive
        try_names = ["Instagram Tag"]
        ig_col = None
        for name in try_names:
            if name in fieldnames:
                ig_col = name
                break
        if ig_col is None:
            # case-insensitive match
            lower_map = {h.lower(): h for h in fieldnames}
            if "instagram tag" in lower_map:
                ig_col = lower_map["instagram tag"]

        if ig_col is None:
            print("Could not find column 'Instagram Tag' in CSV header.")
            print("Found columns:", fieldnames)
            sys.exit(1)

        rows = list(reader)

    # Ensure output columns exist in fieldnames
    follower_col = "Follower Count"
    posts_col = "Number of Posts"
    if follower_col not in fieldnames:
        fieldnames.append(follower_col)
    if posts_col not in fieldnames:
        fieldnames.append(posts_col)

    # Playwright session
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless, slow_mo=50 if not args.headless else 0)

        ctx_kwargs = dict(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1200, "height": 900},
            java_script_enabled=True,
            locale="en-US",
        )
        if Path(AUTH_STATE).exists():
            ctx_kwargs["storage_state"] = AUTH_STATE
        context = browser.new_context(**ctx_kwargs)

        ensure_logged_in(context, headless=args.headless)
        page = context.new_page()

        # Process rows and write output
        print(">>> Writing to:", out_csv.resolve())
        with out_csv.open("w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
            f_out.flush(); import os; os.fsync(f_out.fileno())                    # force header now
            print(">>> Header written")

            for i, row in enumerate(rows, 1):
                handle = (row.get(ig_col) or "").strip()
                if not handle:
                    row[follower_col] = row.get(follower_col, "")
                    row[posts_col] = row.get(posts_col, "")
                    writer.writerow(row)
                    f_out.flush(); import os; os.fsync(f_out.fileno())
                    continue

                print(f"[{i}/{len(rows)}] @{handle} …")
                followers = posts = None
                try:
                    posts, followers = fetch_one(page, handle)
                except Exception as e:
                    print(f"  error: {type(e).__name__}: {e}")

                # Write values (leave empty if None to avoid junk text in the CSV)
                row[follower_col] = followers if followers is not None else ""
                row[posts_col] = posts if posts is not None else ""
                f_out.flush(); import os; os.fsync(f_out.fileno())
                writer.writerow(row)

                # Small delay to be gentle
                import random, time
                sleep_time = random.uniform(2.5, 6.0)
                print(f"  sleeping {sleep_time:.1f}s...")
                time.sleep(sleep_time)

        try:
            page.close()
        except Exception:
            pass
        browser.close()


if __name__ == "__main__":
    main()
