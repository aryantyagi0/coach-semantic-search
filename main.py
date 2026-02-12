import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Reading Excel...")
df = pd.read_excel("alpha_coach_data.xlsx")
print("Rows in dataset:", len(df))

# OPTIONAL: limit rows for testing (remove later)
# df = df.head(100)

print("Creating search_text...")
df["search_text"] = (
    df["title"].fillna("") + " " +
    df["certifications"].fillna("") + " " +
    df["location"].fillna("") + " " +
    df["name"].fillna("") + " " +
    df["category"].fillna("")
)

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded")

print("Generating embeddings...")
coach_embeddings = model.encode(
    df["search_text"].tolist(),
    batch_size=32,          # Faster batching
    show_progress_bar=True,
    convert_to_numpy=True
)
print("Embeddings created")
def apply_filters(df, filters):
    filtered_df = df.copy()

    if "location" in filters:
        filtered_df = filtered_df[
            filtered_df["location"].str.contains(filters["location"], case=False, na=False)
        ]

    if "verified" in filters:
        filtered_df = filtered_df[
            filtered_df["coach_verified"] == filters["verified"]
        ]

    if "assured" in filters:
        filtered_df = filtered_df[
            filtered_df["is_alphacoach_assured"] == filters["assured"]
        ]

    if "category" in filters:
        filtered_df = filtered_df[
        filtered_df["search_text"].str.contains(filters["category"], case=False, na=False)
    ]


    if "min_experience" in filters:
        filtered_df = filtered_df[
            filtered_df["experience"].fillna(0) >= filters["min_experience"]
        ]

    if "min_clients" in filters:
        filtered_df = filtered_df[
            filtered_df["clients_trained"].fillna(0) >= filters["min_clients"]
        ]

    return filtered_df
import re

import re

def extract_filters(query):
    query_lower = query.lower()
    filters = {}

    # verified filter
    if "verified" in query_lower:
        filters["verified"] = True

    if "assured" in query_lower:
        filters["assured"] = True

    # -------- LOCATION DETECTION (UNIVERSAL) --------
    location_values = df["location"].dropna().astype(str).str.lower().unique()

    for loc in location_values:
        parts = [p.strip() for p in loc.split(",")]
        for part in parts:
            if len(part) > 3 and part in query_lower:
                filters["location"] = part
                break
        if "location" in filters:
            break

    # -------- CATEGORY DETECTION --------
    categories = ["yoga", "fitness", "strength", "zumba", "meditation"]
    for cat in categories:
        if cat in query_lower:
            filters["category"] = cat
            break

    # -------- EXPERIENCE DETECTION --------
    match = re.search(r"(\d+)\s*(year|years)", query_lower)
    if match:
        filters["min_experience"] = int(match.group(1))

    return filters



def semantic_search(query, df, coach_embeddings, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)

    scores = cosine_similarity(query_embedding, coach_embeddings)[0]

    df_copy = df.copy()
    df_copy["score"] = scores

    return df_copy.sort_values("score", ascending=False).head(top_k)

print("Running search...")

query = input("Enter your search query: ")

# Step 1 — extract filters
filters = extract_filters(query)
print("Extracted filters:", filters)

# Step 2 — apply filters
filtered_df = apply_filters(df, filters)
print("Rows after filtering:", len(filtered_df))

# Step 3 — fallback if no rows
if len(filtered_df) == 0:
    print("No rows after filtering — falling back to full dataset")
    filtered_df = df

# Step 4 — embeddings for filtered rows
filtered_embeddings = model.encode(
    filtered_df["search_text"].tolist(),
    convert_to_numpy=True
)

# Step 5 — semantic ranking
results = semantic_search(query, filtered_df, filtered_embeddings)

print("\nTop Results:\n")
print(results[["title", "location", "score"]])
