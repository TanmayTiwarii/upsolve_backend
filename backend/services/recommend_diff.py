import os
from collections import defaultdict

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Dataset — loaded & preprocessed once at startup
# ──────────────────────────────────────────────────────────────────────────────
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace("\n", "_")
                  .str.replace(" ", "_")
    )
    return df

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df["text_features"] = df["difficulty"].astype(str) + " " + df["topics"].astype(str)
    return df

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path     = os.path.join(BASE_DIR, "LeetCode Questions.csv")
df       = preprocess_dataset(load_dataset(path))

# ──────────────────────────────────────────────────────────────────────────────
# Inverted index: tag (lowercase) → frozenset of problem IDs
# Built once at import time — O(N × avg_tags) ≈ a few ms, never repeated.
# ──────────────────────────────────────────────────────────────────────────────
def _build_tag_index(df: pd.DataFrame) -> dict[str, frozenset]:
    index: dict[str, set] = defaultdict(set)
    for _, row in df.iterrows():
        topics_str = row.get("topics", "")
        if not topics_str or pd.isna(topics_str):
            continue
        pid = int(row["id"])
        for tag in str(topics_str).split(","):
            index[tag.strip().lower()].add(pid)
    # freeze so callers cannot mutate shared state
    return {tag: frozenset(ids) for tag, ids in index.items()}

# id → row lookup for O(1) slice at request time
df["id"]       = df["id"].astype(int)
_id_to_row     = df.set_index("id")
_tag_index     = _build_tag_index(df)
_all_tags      = frozenset(_tag_index.keys())

# ──────────────────────────────────────────────────────────────────────────────
# Scoring weights
# ──────────────────────────────────────────────────────────────────────────────
SEEN_TAG_WEIGHT   = 1    # tag the user has already practiced
UNSEEN_TAG_WEIGHT = 100  # tag the user hasn't touched yet


# ──────────────────────────────────────────────────────────────────────────────
# recommend_diff
# ──────────────────────────────────────────────────────────────────────────────
def recommend_diff(df: pd.DataFrame, last_20_ids: list, top_k: int = 5) -> pd.DataFrame:
    """
    1. Build seen_tags from the user's last 20 solved problems.
    2. unseen_tags = all_tags - seen_tags.
    3. candidate_ids = union of tag_index[t] for t in unseen_tags  (200-400 IDs).
    4. Score only those candidates — O(candidates × avg_tags) instead of O(all problems).
    5. Weighted sample top_k.
    """
    solved_ids = frozenset(int(x) for x in last_20_ids)

    # ── Step 1: seen tags from solved problems ────────────────────────────────
    seen_tags: set[str] = set()
    for pid in solved_ids:
        if pid not in _id_to_row.index:
            continue
        topics_str = _id_to_row.at[pid, "topics"]
        if topics_str and not pd.isna(topics_str):
            for tag in str(topics_str).split(","):
                seen_tags.add(tag.strip().lower())

    # ── Step 2: unseen tags & candidate pool via inverted index ──────────────
    unseen_tags = _all_tags - seen_tags

    candidate_ids: set[int] = set()
    for tag in unseen_tags:
        candidate_ids |= _tag_index[tag]          # set union, O(1) per tag
    candidate_ids -= solved_ids                    # drop already-solved

    if not candidate_ids:
        # Fallback: user has seen every tag — return any unseen problem
        candidate_ids = set(_id_to_row.index) - solved_ids

    if not candidate_ids:
        return pd.DataFrame(columns=["id", "problem_name", "difficulty", "topics"])

    # ── Step 3: slice DataFrame to candidates only ────────────────────────────
    candidates = _id_to_row.loc[
        _id_to_row.index.intersection(candidate_ids),
        ["problem_name", "difficulty", "topics"]
    ].reset_index()                               # id back as a column

    # ── Step 4: score candidates (only ~200-400 rows, not all 3000) ──────────
    def problem_score(topics_str) -> int:
        if not topics_str or pd.isna(topics_str):
            return UNSEEN_TAG_WEIGHT
        tags = [t.strip().lower() for t in str(topics_str).split(",")]
        return sum(
            SEEN_TAG_WEIGHT if tag in seen_tags else UNSEEN_TAG_WEIGHT
            for tag in tags
        )

    candidates["_score"] = candidates["topics"].apply(problem_score)

    # ── Step 5: weighted sample ───────────────────────────────────────────────
    recs = candidates.sample(
        n=min(top_k, len(candidates)),
        weights="_score",
        random_state=None,          # different result each call
    )

    return recs[["id", "problem_name", "difficulty", "topics"]]