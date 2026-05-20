# ML Backend LeetCode Helper

A Machine Learning-powered REST API built with FastAPI that provides personalized LeetCode problem recommendations based on a user's recent submission history.

## Architecture & Pipeline

The recommendation pipeline ensures high performance and accurate suggestions:

1. **Startup Phase:**
   - Uses FastAPI's `@app.on_event("startup")` to load preprocessed datasets (`processed.csv`) and their precomputed SentenceTransformer vector embeddings (`embeddings.npy`).
   - The `/recommend/diff` service also builds a **tag → problem ID inverted index** at this stage (see [Different Problems](#different-problems-recommenddiff) below).
   - Loading on startup instead of globally allows fast port binding for deployment health checks.

2. **Client Request:**
   - The frontend requests `/recommend/similar` or `/recommend/diff` with the LeetCode username.

3. **Data Fetching (GraphQL):**
   - The backend uses `httpx` to interact with the **LeetCode GraphQL API** (`https://leetcode.com/graphql/`).
   - **Step 1 — Recent submissions:** A single `recentAcSubmissions` query fetches the `titleSlug` of the 20 most recently accepted problems for the given username.
   - **Step 2 — Batched ID lookup:** Rather than firing one `questionData` query per slug (N+1 pattern), all slugs are resolved in a **single batched GraphQL request** using field aliases:
     ```graphql
     query {
       q0: question(titleSlug: "two-sum")      { questionFrontendId }
       q1: question(titleSlug: "rotate-image") { questionFrontendId }
       ...
     }
     ```
   - This reduces total HTTP round-trips from **N+1 down to 2**, regardless of how many unique slugs are returned.

4. **Embedding Preprocessing (Offline):**
   - The dataset of LeetCode problems (`LeetCode Questions.csv`) is preprocessed offline via a dedicated Python script.
   - The text features (specifically, `difficulty` + `topics`, e.g., `"Medium Array Hash Table"`) are concatenated.
   - A lightweight SentenceTransformer model (`paraphrase-MiniLM-L3-v2`) processes these combined strings and encodes them into normalized, high-dimensional dense vectors.
   - These vectors are saved locally as an `embeddings.npy` NumPy array file alongside a cleaned `processed.csv`. This means the API never performs expensive NLP inference during runtime.

5. **Recommendation Engine:**

   ### Similar Problems (`/recommend/similar`)
   - Retrieves the exact embedding vectors for the user's recently solved `questionFrontendId`s from `embeddings.npy`.
   - Calculates the mathematical average of these vectors to generate a single **"User Profile Vector"**.
   - Uses scikit-learn's **Cosine Similarity** metric to measure the angle between the User Profile Vector and every other problem vector in the dataset. Scores closer to 1.0 indicate high semantic overlap in topic and difficulty.
   - Applies a **Hybrid Recommendation Logic** to the sorted list, ensuring the final recommendations match the user's recent Easy/Medium/Hard difficulty ratio.

   ### Different Problems (`/recommend/diff`)
   A tag-weight-based engine designed to surface problems in topics the user has **not** yet practiced.

   **Startup (once):** An inverted index is built from the full dataset:
   ```
   tag (lowercase) → frozenset of problem IDs

   e.g. "dynamic programming" → frozenset({5, 10, 62, 64, ...})
        "graph"               → frozenset({200, 207, 210, ...})
   ```

   **Per request:**
   1. Extract `seen_tags` from the user's last 20 solved problems.
   2. Compute `unseen_tags = all_tags − seen_tags`.
   3. Union the inverted index entries for all `unseen_tags` → **candidate pool of ~200–400 problem IDs** (set union, no full scan).
   4. Score only the candidates using tag weights:
      - **Seen tag** → weight `1` (familiar territory, unlikely to be picked)
      - **Unseen tag** → weight `100` (fresh topic, highly preferred)
      - A problem's score = sum of its tag weights.
   5. Use `pandas.DataFrame.sample(weights=score)` to pick 5 problems — high-scoring problems are proportionally more likely to appear.

   This is ~10× faster than scoring the full 3,000-row dataset, since only problems reachable through unexplored tags are ever evaluated.

6. **Response:**
   - Returns the top 5 customized problem recommendations in JSON format.
   - All floating-point values (`NaN`, `inf`) are sanitized before serialization to prevent JSON encoding errors from missing CSV fields or degenerate embedding vectors.

---

## Technology Stack & Decisions

| Technology | Reason |
|---|---|
| **FastAPI** | Native `async/await`, auto-generated Swagger UI, high throughput vs Flask/Django |
| **SentenceTransformers (`paraphrase-MiniLM-L3-v2`)** | Lightweight BERT-based model for semantic embeddings without GPU |
| **Precomputed Embeddings (`.npy`)** | Offline generation drops inference latency from seconds → milliseconds at runtime |
| **Asynchronous `httpx`** | Non-blocking HTTP client; LeetCode GraphQL calls don't stall the event loop |
| **Batched GraphQL aliases** | Reduces external API round-trips from N+1 to 2 for submission slug resolution |
| **Inverted tag index** | O(1) candidate lookup per tag; limits scoring to ~200–400 rows instead of 3,000 |

---

## Setup Instructions

### Option 1: Local Development (Native Python)

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Mac/Linux
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables:
   - Make sure your `.env` file is set up in the `backend` folder.
5. Run the server:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

### Option 2: Docker Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Build and run using Docker Compose:
   ```bash
   docker compose up --build
   ```
3. The API will be available at `http://localhost:8000`.

---

## Scalability Roadmap

While the current architecture handles the existing 3,000+ LeetCode problems easily, scaling to millions of entries would involve:

* **Vector Databases:** Replacing the in-memory NumPy arrays and O(N) linear Cosine Similarity scan with Approximate Nearest Neighbor (ANN) search using **Pinecone**, **Milvus**, or **FAISS** for O(log N) complexity.
* **Database Migration:** Moving from static CSV files loaded via Pandas to a distributed relational database like **PostgreSQL**.
* **Caching Layer:** Implementing **Redis** to cache LeetCode GraphQL responses and generated recommendations for frequently active users.
* **Persistent Inverted Index:** Storing the tag index in Redis or a search engine like **Elasticsearch** to avoid rebuild cost on cold starts in a multi-worker deployment.
