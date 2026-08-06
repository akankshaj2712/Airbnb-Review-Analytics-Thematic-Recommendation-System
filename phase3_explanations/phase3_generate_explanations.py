"""
PHASE 3 - LLM EXPLANATION LAYER (run in Colab with GPU)
For each listing, generate a one-sentence "why recommended" blurb grounded in its
themes + aspect-sentiment (already in Neon), using a local Ollama model, and cache
the result back into Neon (listings.explanation). The app then just READS the column.

WHY THIS DESIGN:
  - grounded in data already in Neon -> no Phase 0/1 rework
  - precomputed once on Colab GPU -> fast batch, free at serve time, works when deployed
  - resumable: only generates where explanation IS NULL, so a disconnect doesn't lose work

COLAB SETUP (run these in cells BEFORE this script):
  # 1. install + start Ollama, pull a small model
  !curl -fsSL https://ollama.com/install.sh | sh
  import subprocess, time
  subprocess.Popen(["ollama", "serve"]); time.sleep(5)
  !ollama pull llama3.2:3b
  # 2. deps + connection string
  !pip install -q psycopg2-binary requests
  import os; os.environ["NEON_DSN"] = "postgresql://...?sslmode=require"
  # 3. add the column once (or run the ALTER in Neon SQL editor):
  #    ALTER TABLE listings ADD COLUMN IF NOT EXISTS explanation TEXT;
"""

import os
import time
import requests
import psycopg2

DSN          = os.environ["NEON_DSN"]
OLLAMA_URL   = "http://localhost:11434/api/generate"
MODEL        = "llama3.2:3b"
MIN_MENTIONS = 3          # ignore aspects with too few mentions (low confidence)
COMMIT_EVERY = 50

# pct_negative -> plain-language sentiment phrase
def sentiment_phrase(pct_neg):
    if pct_neg >= 0.25:
        return "often criticized"
    if pct_neg >= 0.12:
        return "occasionally noted as an issue"
    return "consistently praised"

def build_prompt(name, themes, aspects):
    theme_str = ", ".join(themes) if themes else "no distinctive themes"
    asp_lines = [f"- {a}: {sentiment_phrase(pn)}" for a, pn in aspects]
    asp_str = "\n".join(asp_lines) if asp_lines else "- (no aspect data)"
    return f"""You are writing a one-sentence recommendation summary for an Airbnb listing.
Use ONLY the facts below. Do not invent anything not listed. Mention one or two standout
positives, and if a drawback is present, mention it honestly. Keep it under 30 words,
natural and specific. Output only the sentence, no preamble.

Listing: {name}
Themes guests mention: {theme_str}
Aspect sentiment:
{asp_str}

One-sentence summary:"""

def generate(prompt):
    r = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 60},
    }, timeout=120)
    r.raise_for_status()
    return r.json()["response"].strip().strip('"')

def main():
    conn = psycopg2.connect(DSN)
    conn.autocommit = False
    cur = conn.cursor()

    # Make sure the column exists
    cur.execute("ALTER TABLE listings ADD COLUMN IF NOT EXISTS explanation TEXT;")
    conn.commit()

    # Pre-fetch all non-generic themes per listing (one query)
    cur.execute("""
        SELECT lt.listing_id, t.label
        FROM listing_themes lt JOIN topics t ON t.topic_id = lt.topic_id
        WHERE t.is_generic = FALSE
        ORDER BY lt.listing_id, lt.weight DESC
    """)
    themes_by_listing = {}
    for lid, label in cur.fetchall():
        themes_by_listing.setdefault(lid, [])
        if len(themes_by_listing[lid]) < 3:
            themes_by_listing[lid].append(label)

    # Pre-fetch aspect sentiment per listing (one query)
    cur.execute("""
        SELECT listing_id, aspect, pct_negative
        FROM listing_aspect_sentiment
        WHERE n_mentions >= %s
        ORDER BY listing_id, n_mentions DESC
    """, [MIN_MENTIONS])
    aspects_by_listing = {}
    for lid, aspect, pct_neg in cur.fetchall():
        aspects_by_listing.setdefault(lid, []).append((aspect, pct_neg or 0))

    # Only listings that still need an explanation (resumable)
    cur.execute("SELECT id, name FROM listings WHERE explanation IS NULL")
    todo = cur.fetchall()
    print(f"{len(todo)} listings need explanations")

    done = 0
    for lid, name in todo:
        themes  = themes_by_listing.get(lid, [])
        aspects = aspects_by_listing.get(lid, [])[:4]
        prompt = build_prompt(name or "this listing", themes, aspects)
        try:
            text = generate(prompt)
        except Exception as e:
            print(f"  skip {lid}: {e}")
            continue
        cur.execute("UPDATE listings SET explanation = %s WHERE id = %s", [text, lid])
        done += 1
        if done % COMMIT_EVERY == 0:
            conn.commit()
            print(f"  {done}/{len(todo)} done")
    conn.commit()

    cur.execute("SELECT count(*) FROM listings WHERE explanation IS NOT NULL")
    total = cur.fetchone()[0]
    cur.close(); conn.close()
    print(f"Done. {done} generated this run; {total} listings now have explanations.")

if __name__ == "__main__":
    main()
