"""
PHASE 2 - SERVING APP (Streamlit)  [v3: review-count confidence weighting]
Free-text semantic search + aspect-sentiment + similar listings, over Neon/pgvector.

RANKING (both search modes):
  score = ( (1 - distance)  s                                # semantic similarity
            + alpha * aspect_sentiment * mention_confidence # sentiment boost (if aspect detected)
          ) * review_confidence                             # down-weight thinly-reviewed listings
  review_confidence = ln(1 + min(n_reviews, CAP)) / ln(1 + CAP)   # saturating log curve
  plus a soft floor: listings with < MIN_REVIEWS are excluded.

Other features:
  - query routed to one of 8 aspects (Phase-0 embedding routing); blends sentiment when matched
  - retrieve-then-rerank (top-N by distance, then re-rank by blended score)
  - badges grey out aspects with < MIN_MENTIONS and show the count
  - generic topics excluded from displayed themes (t.is_generic = FALSE)

RUN:
    pip install streamlit psycopg2-binary pgvector sentence-transformers pandas numpy
    # .streamlit/secrets.toml -> NEON_DSN = "postgresql://...?sslmode=require"
    streamlit run app.py
"""

import numpy as np
import psycopg2
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from pgvector.psycopg2 import register_vector

st.set_page_config(page_title="Airbnb Theme Recommender", page_icon="🏠", layout="wide")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # must match Phase 0
TOP_K = 12
CANDIDATES = 50  # retrieve-then-rerank pool size
MIN_MENTIONS = 3  # below this, an aspect badge is low-confidence
MIN_REVIEWS = 5  # soft floor: exclude listings thinner than this
REVIEW_CAP = 150  # review count beyond which confidence saturates at 1.0
ASPECT_MATCH_THRESHOLD = 0.32  # query->aspect routing cutoff

ASPECT_SEEDS = {
    "cleanliness": ["clean", "dirty", "spotless", "tidy", "messy", "dusty", "hygiene"],
    "location": [
        "location",
        "neighborhood",
        "walkable",
        "close to",
        "far from",
        "transit",
        "downtown",
    ],
    "host": [
        "host",
        "communication",
        "responsive",
        "helpful",
        "check in",
        "friendly",
        "welcoming",
    ],
    "value": [
        "value",
        "price",
        "worth",
        "expensive",
        "cheap",
        "overpriced",
        "affordable",
    ],
    "comfort": ["comfortable", "bed", "cozy", "spacious", "cramped", "small", "comfy"],
    "noise": ["noise", "quiet", "loud", "noisy", "traffic", "peaceful", "thin walls"],
    "accuracy": [
        "as described",
        "accurate",
        "photos",
        "misleading",
        "expected",
        "different from",
    ],
    "amenities": [
        "kitchen",
        "wifi",
        "parking",
        "amenities",
        "shower",
        "heating",
        "ac",
        "appliances",
    ],
}

# Reusable SQL fragment: saturating log confidence from listings.number_of_reviews
REVIEW_CONF = f"(LN(1 + LEAST(COALESCE(c.number_of_reviews, 0), {REVIEW_CAP})) / LN(1 + {REVIEW_CAP}))"


# ------------------------------------------------------------------ resources
@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def get_conn():
    conn = psycopg2.connect(st.secrets["NEON_DSN"])
    conn.autocommit = True
    register_vector(conn)
    return conn


@st.cache_resource
def get_aspect_centroids():
    emb = get_embedder()
    names = list(ASPECT_SEEDS.keys())
    mat = np.vstack(
        [
            emb.encode(seeds, normalize_embeddings=True).mean(axis=0)
            for seeds in ASPECT_SEEDS.values()
        ]
    )
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    return names, mat


@st.cache_data
def get_neighbourhoods():
    with get_conn().cursor() as cur:
        cur.execute(
            "SELECT DISTINCT neighbourhood_cleansed FROM listings "
            "WHERE neighbourhood_cleansed IS NOT NULL ORDER BY 1"
        )
        return [r[0] for r in cur.fetchall()]


@st.cache_data
def get_room_types():
    with get_conn().cursor() as cur:
        cur.execute(
            "SELECT DISTINCT room_type FROM listings "
            "WHERE room_type IS NOT NULL ORDER BY 1"
        )
        return [r[0] for r in cur.fetchall()]


# ------------------------------------------------------------------ embeddings / routing
def embed_query(text):
    return get_embedder().encode(
        "Represent this sentence for retrieving relevant passages: " + text,
        normalize_embeddings=True,
    )


def match_aspect(text):
    names, mat = get_aspect_centroids()
    qv = get_embedder().encode(text, normalize_embeddings=True)
    sims = mat @ qv
    i = int(sims.argmax())
    return (
        (names[i], float(sims[i]))
        if sims[i] >= ASPECT_MATCH_THRESHOLD
        else (None, float(sims[i]))
    )


# ------------------------------------------------------------------ data access
def _filters(neighbourhoods, price_max, room_types):
    where, params = [
        "e.embedding IS NOT NULL",
        f"COALESCE(l.number_of_reviews, 0) >= {MIN_REVIEWS}",
    ], []
    if neighbourhoods:
        where.append("l.neighbourhood_cleansed = ANY(%s)")
        params.append(neighbourhoods)
    if room_types:
        where.append("l.room_type = ANY(%s)")
        params.append(room_types)
    if price_max:
        where.append("l.price <= %s")
        params.append(price_max)
    return " AND ".join(where), params


def search(
    query_vec, neighbourhoods, price_max, room_types, k=TOP_K, aspect=None, alpha=0.0
):
    where, fparams = _filters(neighbourhoods, price_max, room_types)

    if not (aspect and alpha > 0):
        # pure semantic search, weighted by review confidence
        sql = f"""
            WITH candidates AS (
                SELECT l.id, l.name, l.neighbourhood_cleansed, l.room_type, l.price,
                       l.review_scores_rating, l.number_of_reviews, l.listing_url, l.picture_url,
                       e.embedding <=> %s AS distance
                FROM listing_embeddings e JOIN listings l ON l.id = e.listing_id
                WHERE {where}
                ORDER BY distance LIMIT %s
            )
            SELECT c.*, (1 - c.distance) * {REVIEW_CONF} AS score
            FROM candidates c
            ORDER BY score DESC LIMIT %s
        """
        params = [query_vec, *fparams, CANDIDATES, k]
    else:
        # blended: similarity + aspect sentiment, all weighted by review confidence
        sql = f"""
            WITH candidates AS (
                SELECT l.id, l.name, l.neighbourhood_cleansed, l.room_type, l.price,
                       l.review_scores_rating, l.number_of_reviews, l.listing_url, l.picture_url,
                       e.embedding <=> %s AS distance
                FROM listing_embeddings e JOIN listings l ON l.id = e.listing_id
                WHERE {where}
                ORDER BY distance LIMIT %s
            )
            SELECT c.*,
                   ( (1 - c.distance)
                     + %s * COALESCE(a.mean_sentiment, 0)
                          * (LEAST(COALESCE(a.n_mentions, 0), 5) / 5.0) )
                   * {REVIEW_CONF} AS score
            FROM candidates c
            LEFT JOIN listing_aspect_sentiment a
                   ON a.listing_id = c.id AND a.aspect = %s
            ORDER BY score DESC LIMIT %s
        """
        params = [query_vec, *fparams, CANDIDATES, alpha, aspect, k]

    with get_conn().cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def similar_to(listing_id, k=TOP_K):
    # also apply the review floor so "similar" results aren't thin noise
    sql = f"""
        WITH target AS (SELECT embedding FROM listing_embeddings WHERE listing_id = %s)
        SELECT l.id, l.name, l.neighbourhood_cleansed, l.room_type, l.price,
               l.review_scores_rating, l.number_of_reviews, l.listing_url, l.picture_url,
               e.embedding <=> (SELECT embedding FROM target) AS distance
        FROM listing_embeddings e JOIN listings l ON l.id = e.listing_id
        WHERE e.listing_id <> %s AND COALESCE(l.number_of_reviews, 0) >= {MIN_REVIEWS}
        ORDER BY distance LIMIT %s
    """
    with get_conn().cursor() as cur:
        cur.execute(sql, [listing_id, listing_id, k])
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def aspect_sentiment(listing_id):
    with get_conn().cursor() as cur:
        cur.execute(
            """SELECT aspect, mean_sentiment, pct_negative, n_mentions
                       FROM listing_aspect_sentiment WHERE listing_id = %s
                       ORDER BY n_mentions DESC""",
            [listing_id],
        )
        return cur.fetchall()


def top_themes(listing_id, k=3):
    with get_conn().cursor() as cur:
        cur.execute(
            """SELECT t.label, lt.weight
                       FROM listing_themes lt JOIN topics t ON t.topic_id = lt.topic_id
                       WHERE lt.listing_id = %s AND t.is_generic = FALSE
                       ORDER BY lt.weight DESC LIMIT %s""",
            [listing_id, k],
        )
        return cur.fetchall()


# ------------------------------------------------------------------ UI helpers
def sentiment_badges(listing_id):
    rows = aspect_sentiment(listing_id)
    if not rows:
        return
    chips = []
    for aspect, mean_s, pct_neg, n in rows:
        pct_neg, n = (pct_neg or 0), (n or 0)
        if n < MIN_MENTIONS:
            color = "#9aa0a6"
        elif pct_neg >= 0.25:
            color = "#c0392b"
        elif pct_neg >= 0.12:
            color = "#e67e22"
        else:
            color = "#27ae60"
        chips.append(
            f"<span style='background:{color};color:white;padding:2px 8px;"
            f"border-radius:10px;margin:2px;font-size:0.75rem;display:inline-block'>"
            f"{aspect}: {int(pct_neg*100)}% neg ({n})</span>"
        )
    st.markdown(" ".join(chips), unsafe_allow_html=True)


def get_explanation(listing_id):
    with get_conn().cursor() as cur:
        cur.execute("SELECT explanation FROM listings WHERE id = %s", [listing_id])
        row = cur.fetchone()
        return row[0] if row else None


def render_card(row):
    with st.container(border=True):
        c1, c2 = st.columns([1, 3])
        with c1:
            if row.get("picture_url"):
                st.image(row["picture_url"], use_container_width=True)
        with c2:
            st.markdown(f"**[{row['name']}]({row.get('listing_url','#')})**")
            price = f"${row['price']:.0f}" if pd.notnull(row.get("price")) else "—"
            rating = (
                f"{row['review_scores_rating']:.2f}"
                if pd.notnull(row.get("review_scores_rating"))
                else "—"
            )
            st.caption(
                f"{row['neighbourhood_cleansed']} · {row['room_type']} · {price}/night · "
                f"⭐ {rating} · {row.get('number_of_reviews','?')} reviews · "
                f"similarity {1 - row['distance']:.2f}"
            )
            themes = top_themes(row["id"])
            if themes:
                st.caption("Themes: " + ", ".join(t[0] for t in themes))
            sentiment_badges(row["id"])
            exp = get_explanation(row["id"])
            if exp:
                st.markdown(f"💬 *{exp}*")
            if st.button("Find similar", key=f"sim_{row['id']}"):
                st.session_state["similar_to"] = int(row["id"])
                st.session_state["similar_name"] = row["name"]
                st.rerun()


# ------------------------------------------------------------------ app
st.title("🏠 Airbnb Theme & Sentiment Recommender")
st.caption(
    "Free-text semantic search · aspect-sentiment from ABSA · review-count weighted · pgvector"
)

with st.sidebar:
    st.header("Filters")
    nbhd = st.multiselect("Neighborhood", get_neighbourhoods())
    rooms = st.multiselect("Room type", get_room_types())
    price_max = st.slider("Max price / night ($)", 0, 1000, 1000, step=25)
    price_max = None if price_max == 1000 else price_max
    st.divider()
    sentiment_aware = st.checkbox(
        "Sentiment-aware ranking",
        value=True,
        help="If the query is about an aspect (e.g. cleanliness), boost listings praised for it.",
    )
    alpha = st.slider(
        "Sentiment weight", 0.0, 0.5, 0.20, 0.05, disabled=not sentiment_aware
    )
    st.caption(
        f"Listings with < {MIN_REVIEWS} reviews are excluded; "
        f"more reviews = higher confidence (saturating at {REVIEW_CAP})."
    )

if st.session_state.get("similar_to"):
    lid = st.session_state["similar_to"]
    st.subheader(f"Listings similar to: {st.session_state.get('similar_name','')}")
    if st.button("← Back to search"):
        st.session_state.pop("similar_to", None)
        st.rerun()
    for _, row in similar_to(lid).iterrows():
        render_card(row.to_dict())
else:
    query = st.text_input(
        "Describe what you want", placeholder="e.g. clean quiet place near the water"
    )
    if query:
        aspect, sim = match_aspect(query) if sentiment_aware else (None, 0.0)
        with st.spinner("Searching..."):
            qvec = embed_query(query)
            df = search(
                qvec,
                nbhd,
                price_max,
                rooms,
                aspect=aspect,
                alpha=(alpha if sentiment_aware else 0.0),
            )
        if aspect:
            st.success(
                f"Detected aspect **{aspect}** — boosting listings guests praise for it "
                f"(match {sim:.2f}). Toggle off for pure semantic ranking."
            )
        else:
            st.caption(
                "No specific aspect detected — ranking by semantic similarity (review-weighted)."
            )
        if df.empty:
            st.info("No matches with those filters — try widening them.")
        else:
            st.caption(f"{len(df)} results")
            for _, row in df.iterrows():
                render_card(row.to_dict())
    else:
        st.info("Type a description above to get recommendations.")

# .streamlit/secrets.toml  ->  NEON_DSN = "postgresql://...?sslmode=require"   (do NOT commit)
