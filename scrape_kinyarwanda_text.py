#!/usr/bin/env python3
"""
Scrape Kinyarwanda text from public sources to expand the LM corpus.

Sources:
  1. Kinyarwanda Wikipedia (rw.wikipedia.org) — MediaWiki API
  2. Igihe.com — news articles
  3. KigaliToday.com — news articles

Output: appends new text to lm_corpus.txt (one sentence per line)
"""

import os
import re
import sys
import json
import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup

PROJECT_DIR = Path(__file__).parent
CORPUS_PATH = PROJECT_DIR / "lm_corpus.txt"
NEW_PATH    = PROJECT_DIR / "lm_corpus_new.txt"   # intermediate file for new text

HEADERS = {"User-Agent": "KinyarwandaLM/1.0 (lm expansion bot; kinyarwanda-stt)"}

# ── Helpers ──────────────────────────────────────────────────────────────────

def already_has(text: str, existing: set) -> bool:
    """Check if a normalized sentence already exists in the corpus."""
    norm = text.strip().lower()
    return norm in existing


def load_existing_sentences(path: Path) -> set[str]:
    """Load existing sentences for dedup."""
    if not path.exists():
        return set()
    existing = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if line:
            existing.add(line)
    return existing


def save_sentences(sentences: list[str], path: Path, existing: set[str]):
    """Append new unique sentences to file, one per line."""
    count = 0
    with open(path, "a", encoding="utf-8") as f:
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if already_has(s, existing):
                continue
            f.write(s + "\n")
            existing.add(s.lower())
            count += 1
    return count


def clean_text(text: str) -> str:
    """Clean extracted text into sentence-per-line format."""
    # Remove reference markers like [1], [citation needed], etc.
    text = re.sub(r"\[[\w\s]+\]", "", text)
    # Remove multiple spaces / newlines
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    # Split into sentences (simple heuristic: ., !, ?)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned = []
    for s in sentences:
        s = s.strip()
        if len(s) < 10:  # skip very short fragments
            continue
        # Skip lines with too much non-Kinyarwanda content (e.g., English)
        if has_few_english_words(s):
            cleaned.append(s)
    return "\n".join(cleaned)


def has_few_english_words(text: str) -> bool:
    """Return True if text is mostly Kinyarwanda (not English)."""
    # Simple heuristic: if >30% of words contain non-Rwandan patterns, skip
    words = text.split()
    if not words:
        return False
    # Common English function words that shouldn't appear in Kinyarwanda text
    eng_markers = {"the", "and", "for", "are", "was", "were", "been",
                   "has", "had", "have", "from", "they", "that", "this",
                   "which", "with", "what", "when", "where", "how"}
    eng_count = sum(1 for w in words if w.lower() in eng_markers)
    # If >20% of words are English function words, skip
    if eng_count / len(words) > 0.20:
        return False
    return True


# ── 1. Wikipedia Scraper ─────────────────────────────────────────────────────

def scrape_wikipedia(existing: set[str]) -> int:
    """Scrape all Kinyarwanda Wikipedia articles via the MediaWiki API."""
    print("=" * 60)
    print("Scraping Kinyarwanda Wikipedia (rw.wikipedia.org)...")
    api_url = "https://rw.wikipedia.org/w/api.php"
    
    total_added = 0
    page_titles = []
    
    # Step 1: Get all page titles
    print("  Listing all page titles...")
    apfrom = ""
    while True:
        params = {
            "action": "query",
            "format": "json",
            "generator": "allpages",
            "gaplimit": "500",
            "gapfilterredir": "nonredirects",
            "prop": "info",
        }
        if apfrom:
            params["gapfrom"] = apfrom
        
        try:
            resp = requests.get(api_url, params=params, headers=HEADERS, timeout=30)
            data = resp.json()
        except Exception as e:
            print(f"  API error: {e}")
            break
        
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            break
        
        for pid, page in pages.items():
            title = page.get("title", "")
            if title and not title.startswith("MediaWiki:"):
                page_titles.append(title)
        
        continue_param = data.get("continue", {})
        apfrom = continue_param.get("gapcontinue", "")
        if not apfrom:
            break
        time.sleep(0.5)  # rate limiting
    
    print(f"  Found {len(page_titles)} articles total")
    
    # Step 2: Fetch content in batches
    batch_size = 50
    for i in range(0, len(page_titles), batch_size):
        batch = page_titles[i:i+batch_size]
        params = {
            "action": "query",
            "format": "json",
            "titles": "|".join(batch),
            "prop": "extracts",
            "explaintext": "1",
            "exlimit": str(batch_size),
        }
        
        try:
            resp = requests.get(api_url, params=params, headers=HEADERS, timeout=30)
            data = resp.json()
        except Exception as e:
            print(f"  Error fetching batch {i}: {e}")
            continue
        
        pages = data.get("query", {}).get("pages", {})
        sentences_to_add = []
        for pid, page in pages.items():
            extract = page.get("extract", "")
            if extract:
                cleaned = clean_text(extract)
                if cleaned:
                    for line in cleaned.split("\n"):
                        line = line.strip()
                        if line and len(line) >= 10 and not already_has(line, existing):
                            sentences_to_add.append(line)
        
        added = save_sentences(sentences_to_add, NEW_PATH, existing)
        total_added += added
        
        if (i // batch_size) % 5 == 0:
            print(f"  Progress: {min(i+batch_size, len(page_titles))}/{len(page_titles)} "
                  f"articles, {total_added} new sentences so far")
        
        time.sleep(0.3)  # rate limiting
    
    print(f"  Wikipedia: added {total_added} new sentences")
    return total_added


# ── 2. News Site Scraper ─────────────────────────────────────────────────────

def scrape_igihe(existing: set[str]) -> int:
    """Scrape article text from igihe.com."""
    print("\nScraping Igihe.com news articles...")
    total_added = 0
    
    # Igihe.com categories
    categories = [
        "https://igihe.com/amakuru",
        "https://igihe.com/amakuru/u-rwanda",
        "https://igihe.com/amakuru/africa",
        "https://igihe.com/amakuru/amerika",
        "https://igihe.com/amakuru/asiya",
        "https://igihe.com/amakuru/uburaya",
        "https://igihe.com/amakuru/ibindi",
        "https://igihe.com/imikino",
        "https://igihe.com/ubukungu",
        "https://igihe.com/ubuzima",
        "https://igihe.com/imyidagaduro",
        "https://igihe.com/amateka",
        "https://igihe.com/iteknoloji",
        "https://igihe.com/urugendo",
    ]
    
    for cat_url in categories:
        print(f"  Fetching article list: {cat_url}")
        try:
            resp = requests.get(cat_url, headers=HEADERS, timeout=30)
            soup = BeautifulSoup(resp.text, "html.parser")
            # Find article links
            links = soup.find_all("a", href=True)
            article_urls = set()
            for a in links:
                href = a["href"]
                if "/amakuru/" in href or "/imikino/" in href or "/ubukungu/" in href \
                   or "/ubuzima/" in href or "/imyidagaduro/" in href or "/amateka/" in href \
                   or "/iteknoloji/" in href or "/urugendo/" in href:
                    if href.startswith("/"):
                        href = "https://igihe.com" + href
                    if href not in article_urls:
                        article_urls.add(href)
            
            # Limit to first 20 articles per category
            for url in list(article_urls)[:20]:
                text = fetch_article_text_igihe(url)
                if text:
                    cleaned = clean_text(text)
                    if cleaned:
                        sentences = [s.strip() for s in cleaned.split("\n") if s.strip()]
                        added = save_sentences(sentences, NEW_PATH, existing)
                        total_added += added
                time.sleep(0.5)  # rate limiting
        except Exception as e:
            print(f"    Error with {cat_url}: {e}")
    
    print(f"  Igihe.com: added {total_added} new sentences")
    return total_added


def fetch_article_text_igihe(url: str) -> str | None:
    """Fetch the main article text from an igihe.com article."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove scripts, styles, nav
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        # Look for article body
        article = soup.find("article") or soup.find("div", class_="article-content") or \
                  soup.find("div", class_="content") or soup.find("div", class_="post-content") or \
                  soup.find("div", itemprop="articleBody")
        if article:
            return article.get_text(separator=" ", strip=True)
        # Fallback: body text
        body = soup.find("body")
        if body:
            return body.get_text(separator=" ", strip=True)
        return None
    except Exception:
        return None


def scrape_kigalitoday(existing: set[str]) -> int:
    """Scrape article text from kigalitoday.com."""
    print("\nScraping KigaliToday.com news articles...")
    total_added = 0
    
    categories = [
        "https://www.kigalitoday.com/amakuru",
        "https://www.kigalitoday.com/amakuru/amakuru-mu-rwanda",
        "https://www.kigalitoday.com/amakuru/amakuru-mu-afrika",
        "https://www.kigalitoday.com/amakuru/amakuru-mu-isi",
        "https://www.kigalitoday.com/imikino",
        "https://www.kigalitoday.com/ubuhinzi",
        "https://www.kigalitoday.com/ubukungu",
        "https://www.kigalitoday.com/uburezi",
        "https://www.kigalitoday.com/ubuzima",
        "https://www.kigalitoday.com/imyidagaduro",
        "https://www.kigalitoday.com/iteknoloji",
    ]
    
    for cat_url in categories:
        print(f"  Fetching article list: {cat_url}")
        try:
            resp = requests.get(cat_url, headers=HEADERS, timeout=30)
            soup = BeautifulSoup(resp.text, "html.parser")
            links = soup.find_all("a", href=True)
            article_urls = set()
            for a in links:
                href = a["href"]
                if "/amakuru/" in href or "/imikino/" in href or "/ubuhinzi/" in href \
                   or "/ubukungu/" in href or "/uburezi/" in href or "/ubuzima/" in href \
                   or "/imyidagaduro/" in href or "/iteknoloji/" in href:
                    if href.startswith("/"):
                        href = "https://www.kigalitoday.com" + href
                    if href not in article_urls:
                        article_urls.add(href)
            
            for url in list(article_urls)[:15]:
                text = fetch_article_text_kigalitoday(url)
                if text:
                    cleaned = clean_text(text)
                    if cleaned:
                        sentences = [s.strip() for s in cleaned.split("\n") if s.strip()]
                        added = save_sentences(sentences, NEW_PATH, existing)
                        total_added += added
                time.sleep(0.5)
        except Exception as e:
            print(f"    Error with {cat_url}: {e}")
    
    print(f"  KigaliToday: added {total_added} new sentences")
    return total_added


def fetch_article_text_kigalitoday(url: str) -> str | None:
    """Fetch the main article text from kigalitoday.com."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        article = soup.find("article") or soup.find("div", class_="article-content") or \
                  soup.find("div", class_="content") or soup.find("div", itemprop="articleBody")
        if article:
            return article.get_text(separator=" ", strip=True)
        body = soup.find("body")
        if body:
            return body.get_text(separator=" ", strip=True)
        return None
    except Exception:
        return None


# ── 3. RBA Scraper ───────────────────────────────────────────────────────────

def scrape_rba(existing: set[str]) -> int:
    """Scrape article text from rba.co.rw."""
    print("\nScraping RBA (rba.co.rw) news articles...")
    total_added = 0
    
    categories = [
        "https://www.rba.co.rw/amakuru",
        "https://www.rba.co.rw/amakuru/mu-rwanda",
        "https://www.rba.co.rw/amakuru/mu-isi",
        "https://www.rba.co.rw/siporo",
        "https://www.rba.co.rw/ubukungu",
        "https://www.rba.co.rw/ubuzima",
        "https://www.rba.co.rw/uburezi",
        "https://www.rba.co.rw/umuco",
    ]
    
    for cat_url in categories:
        print(f"  Fetching article list: {cat_url}")
        try:
            resp = requests.get(cat_url, headers=HEADERS, timeout=30)
            soup = BeautifulSoup(resp.text, "html.parser")
            links = soup.find_all("a", href=True)
            article_urls = set()
            for a in links:
                href = a["href"]
                if "/amakuru/" in href or "/siporo/" in href or "/ubukungu/" in href \
                   or "/ubuzima/" in href or "/uburezi/" in href or "/umuco/" in href:
                    if href.startswith("/"):
                        href = "https://www.rba.co.rw" + href
                    if href not in article_urls:
                        article_urls.add(href)
            
            for url in list(article_urls)[:15]:
                try:
                    resp2 = requests.get(url, headers=HEADERS, timeout=30)
                    soup2 = BeautifulSoup(resp2.text, "html.parser")
                    for tag in soup2(["script", "style", "nav", "header", "footer", "aside"]):
                        tag.decompose()
                    article = soup2.find("article") or soup2.find("div", class_="content") or \
                              soup2.find("div", itemprop="articleBody")
                    if article:
                        text = article.get_text(separator=" ", strip=True)
                        cleaned = clean_text(text)
                        if cleaned:
                            sentences = [s.strip() for s in cleaned.split("\n") if s.strip()]
                            added = save_sentences(sentences, NEW_PATH, existing)
                            total_added += added
                    time.sleep(0.5)
                except Exception:
                    continue
        except Exception as e:
            print(f"    Error with {cat_url}: {e}")
    
    print(f"  RBA: added {total_added} new sentences")
    return total_added


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Load existing sentences for dedup
    existing = load_existing_sentences(CORPUS_PATH)
    print(f"Existing corpus: {len(existing)} unique sentences")
    
    # Create new corpus file if it doesn't exist
    if not NEW_PATH.exists():
        NEW_PATH.write_text("", encoding="utf-8")
    
    total = 0
    
    # Source 1: Wikipedia
    total += scrape_wikipedia(existing)
    
    # Source 2: Igihe.com
    total += scrape_igihe(existing)
    
    # Source 3: KigaliToday
    total += scrape_kigalitoday(existing)
    
    # Source 4: RBA
    total += scrape_rba(existing)
    
    # Now merge new text into lm_corpus.txt
    if NEW_PATH.exists() and NEW_PATH.stat().st_size > 0:
        print(f"\n{'='*60}")
        print(f"Total new sentences collected: {total}")
        print(f"Merging into {CORPUS_PATH}...")
        
        # Read existing corpus
        existing_lines = CORPUS_PATH.read_text(encoding="utf-8").splitlines() if CORPUS_PATH.exists() else []
        
        # Read new sentences
        new_lines = NEW_PATH.read_text(encoding="utf-8").splitlines()
        
        # Deduplicate (case-insensitive)
        existing_set = set(l.strip().lower() for l in existing_lines if l.strip())
        merged = list(existing_lines)
        for line in new_lines:
            line = line.strip()
            if line and line.lower() not in existing_set:
                merged.append(line)
                existing_set.add(line.lower())
        
        # Write merged corpus
        CORPUS_PATH.write_text("\n".join(merged) + "\n", encoding="utf-8")
        
        # Clean up temp file
        NEW_PATH.unlink()
        
        print(f"Merged corpus: {len(merged)} lines (was {len(existing_lines)})")
        print(f"Added {len(merged) - len(existing_lines)} new lines")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
