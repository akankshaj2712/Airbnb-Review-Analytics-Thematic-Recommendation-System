# ==============================================================================
# PHASE 1 - DATA LAYER : load Phase 0 parquet artifacts into Neon (Postgres + pgvector)
#
# WHERE TO RUN: anywhere with internet + the 5 parquet files. Easiest is right
#   in Colab after Phase 0 (files already in /content/phase0_out). Or locally.
#
# SETUP (once):
#   1. Create a free Neon project at neon.tech (Postgres 16).
#   2. Copy its connection string (Dashboard -> Connection Details).
#   3. Put it in the NEON_DSN env var (don't hardcode in a shared notebook):
#        Colab:  import os; os.environ["NEON_DSN"] = "postgresql://...:...@...-pooler.../neondb?sslmode=require"
#        Local:  set it in your shell, or edit DSN below.
#
# INSTALL:
#   pip install psycopg2-binary pgvector pandas pyarrow
#
# COLUMN MAPPINGS (parquet -> table), handled below so nothing mismatches:
#   listing_themes:           topic -> topic_id,  count -> review_count
#   listing_aspect_sentiment: n     -> n_mentions
#   listing_embeddings:       cols "0".."383"    -> embedding vector(384)
# ==============================================================================

import os
import math
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values, Json
from pgvector.psycopg2 import register_vector

# --- Config -------------------------------------------------------------------
DSN     = os.environ.get("NEON_DSN", "postgresql://USER:PASSWORD@HOST/neondb?sslmode=require")
OUT_DIR = os.environ.get("PHASE0_OUT", "/content/phase0_out")   # where the 5 parquet files live
EMB_DIM = 384                                                   # bge-small-en
RESET   = True   # drop & recreate tables for a clean load (safe in dev)

# --- Schema -------------------------------------------------------------------
DDL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

{"DROP TABLE IF EXISTS listing_embeddings, listing_aspect_sentiment, listing_themes, listings, topics CASCADE;" if RESET else ""}

CREATE TABLE IF NOT EXISTS topics (
    topic_id   INTEGER PRIMARY KEY,
    label      TEXT,
    top_words  TEXT,
    is_generic BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS listings (
    id                            BIGINT PRIMARY KEY,
    name                          TEXT,
    listing_url                   TEXT,
    picture_url                   TEXT,
    latitude                      DOUBLE PRECISION,
    longitude                     DOUBLE PRECISION,
    neighbourhood_cleansed        TEXT,
    neighbourhood_group_cleansed  TEXT,
    room_type                     TEXT,
    property_type                 TEXT,
    accommodates                  INTEGER,
    bedrooms                      REAL,
    beds                          REAL,
    bathrooms_text                TEXT,
    price                         REAL,
    number_of_reviews             INTEGER,
    reviews_per_month             REAL,
    review_scores_rating          REAL,
    review_scores_cleanliness     REAL,
    review_scores_location        REAL,
    review_scores_communication   REAL,
    review_scores_value           REAL,
    host_is_superhost             BOOLEAN,
    amenities                     JSONB,
    description                   TEXT
);

CREATE TABLE IF NOT EXISTS listing_themes (
    listing_id    BIGINT  REFERENCES listings(id),
    topic_id      INTEGER REFERENCES topics(topic_id),
    review_count  INTEGER,
    weight        REAL,
    PRIMARY KEY (listing_id, topic_id)
);

CREATE TABLE IF NOT EXISTS listing_aspect_sentiment (
    listing_id      BIGINT REFERENCES listings(id),
    aspect          TEXT,
    mean_sentiment  REAL,
    n_mentions      INTEGER,
    pct_negative    REAL,
    PRIMARY KEY (listing_id, aspect)
);

CREATE TABLE IF NOT EXISTS listing_embeddings (
    listing_id  BIGINT PRIMARY KEY REFERENCES listings(id),
    embedding   vector({EMB_DIM})
);
"""

# Indexes built AFTER bulk load (faster). Filter indexes power the structured
# WHERE clauses; the HNSW index powers vector similarity search.
INDEXES = """
CREATE INDEX IF NOT EXISTS ix_listings_neigh    ON listings (neighbourhood_cleansed);
CREATE INDEX IF NOT EXISTS ix_listings_group    ON listings (neighbourhood_group_cleansed);
CREATE INDEX IF NOT EXISTS ix_listings_price    ON listings (price);
CREATE INDEX IF NOT EXISTS ix_listings_roomtype ON listings (room_type);
CREATE INDEX IF NOT EXISTS ix_themes_topic      ON listing_themes (topic_id, weight DESC);
CREATE INDEX IF NOT EXISTS ix_aspect            ON listing_aspect_sentiment (aspect);
CREATE INDEX IF NOT EXISTS ix_emb_hnsw          ON listing_embeddings USING hnsw (embedding vector_cosine_ops);
"""

# --- Helpers ------------------------------------------------------------------
def _clean(v):
    """pandas NaN / NaT -> SQL NULL; leave everything else intact."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v

def _bool_or_none(v):
    return True if v is True else (False if v is False else None)

def _amenities(v):
    """Phase 0 parsed amenities into a list; store as JSONB (or NULL)."""
    if isinstance(v, (list, np.ndarray)):
        return Json(list(v))
    return None

def load_scalar(cur, df, table, colmap):
    """Generic loader for tables of plain scalars. colmap: {parquet_col: table_col}."""
    df2 = df[list(colmap.keys())].rename(columns=colmap)
    df2 = df2.astype(object).where(pd.notnull(df2), None)   # NaN -> None
    rows = list(df2.itertuples(index=False, name=None))
    cols = ", ".join(colmap.values())
    execute_values(cur, f"INSERT INTO {table} ({cols}) VALUES %s", rows, page_size=1000)
    return len(rows)

def load_listings(cur, df):
    fields = [
        "id", "name", "listing_url", "picture_url", "latitude", "longitude",
        "neighbourhood_cleansed", "neighbourhood_group_cleansed", "room_type",
        "property_type", "accommodates", "bedrooms", "beds", "bathrooms_text",
        "price", "number_of_reviews", "reviews_per_month", "review_scores_rating",
        "review_scores_cleanliness", "review_scores_location",
        "review_scores_communication", "review_scores_value",
        "host_is_superhost", "amenities", "description",
    ]
    rows = []
    for rec in df.to_dict("records"):
        rows.append((
            int(rec["id"]),
            _clean(rec.get("name")), _clean(rec.get("listing_url")), _clean(rec.get("picture_url")),
            _clean(rec.get("latitude")), _clean(rec.get("longitude")),
            _clean(rec.get("neighbourhood_cleansed")), _clean(rec.get("neighbourhood_group_cleansed")),
            _clean(rec.get("room_type")), _clean(rec.get("property_type")),
            _clean(rec.get("accommodates")), _clean(rec.get("bedrooms")), _clean(rec.get("beds")),
            _clean(rec.get("bathrooms_text")), _clean(rec.get("price")),
            _clean(rec.get("number_of_reviews")), _clean(rec.get("reviews_per_month")),
            _clean(rec.get("review_scores_rating")), _clean(rec.get("review_scores_cleanliness")),
            _clean(rec.get("review_scores_location")), _clean(rec.get("review_scores_communication")),
            _clean(rec.get("review_scores_value")),
            _bool_or_none(rec.get("host_is_superhost")),
            _amenities(rec.get("amenities")),
            _clean(rec.get("description")),
        ))
    cols = ", ".join(fields)
    execute_values(cur, f"INSERT INTO listings ({cols}) VALUES %s", rows, page_size=500)
    return len(rows)

def load_embeddings(cur, df):
    emb_cols = sorted([c for c in df.columns if c != "listing_id"], key=lambda c: int(c))
    assert len(emb_cols) == EMB_DIM, f"expected {EMB_DIM} dims, got {len(emb_cols)}"
    ids = df["listing_id"].astype("int64").tolist()
    mat = df[emb_cols].to_numpy(dtype=np.float32)            # (n, 384)
    data = [(ids[i], mat[i]) for i in range(len(ids))]       # ndarray rows adapt to vector
    execute_values(cur,
        "INSERT INTO listing_embeddings (listing_id, embedding) VALUES %s",
        data, page_size=500)
    return len(data)

# --- Main ---------------------------------------------------------------------
def main():
    print("Reading parquet files from", OUT_DIR)
    listings   = pd.read_parquet(f"{OUT_DIR}/listings_clean.parquet")
    topics     = pd.read_parquet(f"{OUT_DIR}/topics.parquet")
    themes     = pd.read_parquet(f"{OUT_DIR}/listing_themes.parquet")
    aspects    = pd.read_parquet(f"{OUT_DIR}/listing_aspect_sentiment.parquet")
    embeddings = pd.read_parquet(f"{OUT_DIR}/listing_embeddings.parquet")

    conn = psycopg2.connect(DSN)
    conn.autocommit = True
    register_vector(conn)                  # enables ndarray/list -> vector adaptation
    conn.autocommit = False
    cur = conn.cursor()

    print("Creating schema ...")
    cur.execute(DDL)
    conn.commit()

    print("Loading topics ...")
    n_topics = load_scalar(cur, topics, "topics",
        {"topic_id": "topic_id", "label": "label",
         "top_words": "top_words", "is_generic": "is_generic"})

    print("Loading listings ...")
    n_listings = load_listings(cur, listings)

    print("Loading listing_themes ...")
    n_themes = load_scalar(cur, themes, "listing_themes",
        {"listing_id": "listing_id", "topic": "topic_id",
         "count": "review_count", "weight": "weight"})

    print("Loading listing_aspect_sentiment ...")
    n_aspect = load_scalar(cur, aspects, "listing_aspect_sentiment",
        {"listing_id": "listing_id", "aspect": "aspect",
         "mean_sentiment": "mean_sentiment", "n": "n_mentions",
         "pct_negative": "pct_negative"})

    print("Loading listing_embeddings ...")
    n_emb = load_embeddings(cur, embeddings)

    conn.commit()

    print("Building indexes (incl. HNSW) ...")
    cur.execute(INDEXES)
    conn.commit()

    # Verify
    counts = {}
    for t in ["topics", "listings", "listing_themes",
              "listing_aspect_sentiment", "listing_embeddings"]:
        cur.execute(f"SELECT count(*) FROM {t}")
        counts[t] = cur.fetchone()[0]

    cur.close()
    conn.close()

    print(f"""
PHASE 1 COMPLETE - loaded into Neon
  topics                    : {counts['topics']:>7,}  (inserted {n_topics})
  listings                  : {counts['listings']:>7,}  (inserted {n_listings})
  listing_themes            : {counts['listing_themes']:>7,}  (inserted {n_themes})
  listing_aspect_sentiment  : {counts['listing_aspect_sentiment']:>7,}  (inserted {n_aspect})
  listing_embeddings        : {counts['listing_embeddings']:>7,}  (inserted {n_emb})
Next: Phase 2 - the Streamlit app querying these tables.
""")

if __name__ == "__main__":
    main()
