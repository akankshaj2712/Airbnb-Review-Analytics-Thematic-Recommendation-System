"""
PHASE 4 - AIRFLOW DAG : airbnb_warehouse_refresh
Orchestrates the local warehouse-refresh pipeline (heavy model FIT stays offline/Colab):

    check_artifacts  ->  load_reference  ->  load_derived  ->  quality_check
    (validate parquet)   (upsert topics+   (reload themes,   (verify Neon row
                          listings)         aspects, embeds)  counts & integrity)

Idempotent: listings/topics are UPSERTed (preserving listings.explanation from Phase 3),
derived tables are truncated + reloaded. Safe to re-run.

MEMORY-SAFE build: this host is small (WSL, ~3.7 GB RAM, Airflow itself uses ~1.6 GB).
- check_artifacts validates via Parquet FOOTER METADATA only (row counts + column names),
  so it never materializes a single frame into pandas.
- load_derived streams the wide embeddings frame in row-group batches instead of building
  the whole N x 384 matrix at once.

SETUP (once, in the airflow venv):
    pip install psycopg2-binary pgvector pandas pyarrow
    airflow variables set NEON_DSN "postgresql://...?sslmode=require"
    # copy the 5 parquet files into ~/airflow/data/
    # drop this file into ~/airflow/dags/
"""

from __future__ import annotations
import os
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.models import Variable

DATA_DIR = os.environ.get("PHASE0_DATA", os.path.expanduser("~/airflow/data"))
EMB_DIM  = 384
FILES = {
    "listings":  "listings_clean.parquet",
    "topics":    "topics.parquet",
    "themes":    "listing_themes.parquet",
    "aspects":   "listing_aspect_sentiment.parquet",
    "embeddings":"listing_embeddings.parquet",
}

default_args = {
    "owner": "akanksha",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}


@dag(
    dag_id="airbnb_warehouse_refresh",
    description="Refresh the Airbnb recommender warehouse (Neon + pgvector) from Phase 0 artifacts",
    schedule="@weekly",           # runs weekly; also triggerable manually
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["airbnb", "recommender", "elt"],
)
def airbnb_warehouse_refresh():

    @task
    def check_artifacts() -> dict:
        """Validate all parquet artifacts using footer metadata only (no data loaded)."""
        import pyarrow.parquet as pq

        counts = {}
        for key, fname in FILES.items():
            path = os.path.join(DATA_DIR, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing artifact: {path}")
            pf = pq.ParquetFile(path)          # reads footer only, not the row data
            n_rows = pf.metadata.num_rows
            if n_rows == 0:
                raise ValueError(f"Empty artifact: {fname}")
            counts[key] = n_rows

        # sanity: embedding width, from schema names only (no matrix in memory)
        emb_schema = pq.read_schema(os.path.join(DATA_DIR, FILES["embeddings"]))
        n_dims = len([n for n in emb_schema.names if n != "listing_id"])
        if n_dims != EMB_DIM:
            raise ValueError(f"Embedding dim {n_dims} != expected {EMB_DIM}")

        print("Artifact validation passed:", counts)
        return counts

    @task
    def load_reference() -> None:
        """Upsert topics + listings (preserves listings.explanation)."""
        import pandas as pd, math, psycopg2
        from psycopg2.extras import execute_values, Json

        dsn = Variable.get("NEON_DSN")
        listings = pd.read_parquet(os.path.join(DATA_DIR, FILES["listings"]))
        topics   = pd.read_parquet(os.path.join(DATA_DIR, FILES["topics"]))

        def clean(v):
            return None if (v is None or (isinstance(v, float) and math.isnan(v))) else v

        conn = psycopg2.connect(dsn); conn.autocommit = False; cur = conn.cursor()

        # ensure tables + explanation column exist (no drops)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS topics (
                topic_id INTEGER PRIMARY KEY, label TEXT, top_words TEXT,
                is_generic BOOLEAN DEFAULT FALSE);
            ALTER TABLE topics ADD COLUMN IF NOT EXISTS is_generic BOOLEAN DEFAULT FALSE;
            ALTER TABLE listings ADD COLUMN IF NOT EXISTS explanation TEXT;
        """)

        # topics upsert
        trows = [(int(r.topic_id), clean(r.label), clean(r.top_words), bool(r.is_generic))
                 for r in topics.itertuples(index=False)]
        execute_values(cur, """
            INSERT INTO topics (topic_id, label, top_words, is_generic) VALUES %s
            ON CONFLICT (topic_id) DO UPDATE SET
                label=EXCLUDED.label, top_words=EXCLUDED.top_words,
                is_generic=EXCLUDED.is_generic
        """, trows)

        # listings upsert (every column EXCEPT explanation, which we preserve)
        cols = ["id","name","listing_url","picture_url","latitude","longitude",
                "neighbourhood_cleansed","neighbourhood_group_cleansed","room_type",
                "property_type","accommodates","bedrooms","beds","bathrooms_text","price",
                "number_of_reviews","reviews_per_month","review_scores_rating",
                "review_scores_cleanliness","review_scores_location",
                "review_scores_communication","review_scores_value","host_is_superhost",
                "amenities","description"]
        rows = []
        for rec in listings.to_dict("records"):
            amen = rec.get("amenities")
            amen = Json(list(amen)) if isinstance(amen, (list, tuple)) or hasattr(amen, "__len__") and not isinstance(amen, str) and amen is not None else None
            sh = rec.get("host_is_superhost")
            sh = True if sh is True else (False if sh is False else None)
            rows.append((int(rec["id"]),
                *[clean(rec.get(c)) for c in cols[1:22]],  # name..review_scores_value
                sh, amen, clean(rec.get("description"))))
        set_clause = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c != "id")
        execute_values(cur, f"""
            INSERT INTO listings ({', '.join(cols)}) VALUES %s
            ON CONFLICT (id) DO UPDATE SET {set_clause}
        """, rows, page_size=500)

        conn.commit(); cur.close(); conn.close()
        print(f"Upserted {len(trows)} topics, {len(rows)} listings")

    @task
    def load_derived() -> None:
        """Truncate + reload themes, aspect sentiment, embeddings (no precious columns there)."""
        import pandas as pd, numpy as np, psycopg2
        import pyarrow.parquet as pq
        from psycopg2.extras import execute_values
        from pgvector.psycopg2 import register_vector

        dsn = Variable.get("NEON_DSN")
        themes  = pd.read_parquet(os.path.join(DATA_DIR, FILES["themes"]))
        aspects = pd.read_parquet(os.path.join(DATA_DIR, FILES["aspects"]))

        # Set autocommit BEFORE any query runs. register_vector() executes a
        # SELECT to look up the pgvector OID, which starts a transaction; changing
        # session settings mid-transaction raises "set_session cannot be used
        # inside a transaction". So configure the session first, then register.
        conn = psycopg2.connect(dsn); conn.autocommit = False
        register_vector(conn); cur = conn.cursor()

        cur.execute("TRUNCATE listing_themes, listing_aspect_sentiment, listing_embeddings;")

        execute_values(cur,
            "INSERT INTO listing_themes (listing_id, topic_id, review_count, weight) VALUES %s",
            list(themes[["listing_id","topic","count","weight"]].itertuples(index=False, name=None)),
            page_size=1000)
        del themes

        execute_values(cur,
            "INSERT INTO listing_aspect_sentiment (listing_id, aspect, mean_sentiment, n_mentions, pct_negative) VALUES %s",
            list(aspects[["listing_id","aspect","mean_sentiment","n","pct_negative"]].itertuples(index=False, name=None)),
            page_size=1000)
        del aspects

        # Stream the wide embeddings frame in row-group batches so the full
        # N x 384 matrix never lives in memory at once.
        emb_path = os.path.join(DATA_DIR, FILES["embeddings"])
        emb_schema = pq.read_schema(emb_path)
        emb_cols = sorted([n for n in emb_schema.names if n != "listing_id"], key=lambda c: int(c))

        pf = pq.ParquetFile(emb_path)
        n_emb = 0
        for batch in pf.iter_batches(batch_size=2000, columns=["listing_id"] + emb_cols):
            bdf = batch.to_pandas()
            ids = bdf["listing_id"].astype("int64").tolist()
            mat = bdf[emb_cols].to_numpy(dtype=np.float32)
            execute_values(cur,
                "INSERT INTO listing_embeddings (listing_id, embedding) VALUES %s",
                [(ids[i], mat[i]) for i in range(len(ids))], page_size=500)
            n_emb += len(ids)
            del bdf, mat, ids

        conn.commit(); cur.close(); conn.close()
        print(f"Reloaded themes, aspects, and {n_emb} embeddings")

    @task
    def quality_check() -> None:
        """Fail the run if the loaded data doesn't meet integrity expectations."""
        import psycopg2
        dsn = Variable.get("NEON_DSN")
        conn = psycopg2.connect(dsn); cur = conn.cursor()

        def scalar(q):
            cur.execute(q); return cur.fetchone()[0]

        n_listings = scalar("SELECT count(*) FROM listings")
        n_emb      = scalar("SELECT count(*) FROM listing_embeddings")
        n_orphans  = scalar("""SELECT count(*) FROM listing_themes lt
                               LEFT JOIN listings l ON l.id = lt.listing_id
                               WHERE l.id IS NULL""")
        problems = []
        if n_listings < 100:
            problems.append(f"too few listings: {n_listings}")
        if n_emb != n_listings:
            problems.append(f"embedding count {n_emb} != listing count {n_listings}")
        if n_orphans > 0:
            problems.append(f"{n_orphans} theme rows reference missing listings")

        cur.close(); conn.close()
        if problems:
            raise ValueError("Quality check FAILED: " + "; ".join(problems))
        print(f"Quality check passed: {n_listings} listings, {n_emb} embeddings, 0 orphans")

    # dependency chain
    check_artifacts() >> load_reference() >> load_derived() >> quality_check()


airbnb_warehouse_refresh()