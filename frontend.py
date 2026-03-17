import streamlit as st
from backend import search_pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Coach Discovery Platform",
    layout="wide"
)

st.title("🏋️ AI Coach Discovery Platform")

st.markdown("""
Search fitness coaches using natural language:

Examples:
- best strength training coach in delhi
- certified yoga trainer who speaks hindi
- online fat loss coach with more than 5 years experience
""")

# ---------------- SIDEBAR OPTIONS ----------------
st.sidebar.header("⚙️ Display Options")

show_structured = st.sidebar.checkbox(
    "Show Extracted Query",
    value=True
)

sort_option = st.sidebar.selectbox(
    "Sort Results By",
    [
        "Hybrid Score (Default)",
        "Experience (High to Low)",
        "Experience (Low to High)"
    ]
)

# ---------------- SEARCH INPUT ----------------
query = st.text_input("🔎 Search Coaches")

# Store results in session (prevents reset issue)
if "results_data" not in st.session_state:
    st.session_state.results_data = None

if "structured_data" not in st.session_state:
    st.session_state.structured_data = None

if "matched_rows" not in st.session_state:
    st.session_state.matched_rows = 0

if "fallback_used" not in st.session_state:
    st.session_state.fallback_used = False


# ---------------- SEARCH BUTTON ----------------
if st.button("Search"):

    if not query.strip():
        st.warning("Please enter a query.")
    else:

        with st.spinner("🚀 Running Hybrid Search..."):

            results, structured_query, matched_rows, fallback_used = search_pipeline(query)

        # Save to session
        st.session_state.results_data = results
        st.session_state.structured_data = structured_query
        st.session_state.matched_rows = matched_rows
        st.session_state.fallback_used = fallback_used


# ---------------- DISPLAY RESULTS ----------------
if st.session_state.results_data is not None:

    results = st.session_state.results_data
    structured_query = st.session_state.structured_data
    matched_rows = st.session_state.matched_rows
    fallback_used = st.session_state.fallback_used

    # # 🔴 Handle out-of-domain error
    # if isinstance(structured_query, dict) and structured_query.get("error"):
    #     st.error("❌ This query is not related to fitness coaching.")
    #     st.stop()

    # ---------------- SUMMARY SECTION ----------------
    st.markdown("## 🔍 Search Summary")

    st.write(f"Matched Rows After Structured Filtering: {matched_rows}")

    if fallback_used:
        st.warning("⚠ No structured matches found. Semantic fallback search was triggered.")
    else:
        st.success("Structured filtering applied successfully.")

    st.write(f"Final Results Returned: {len(results)}")

    # ---------------- SHOW STRUCTURED QUERY ----------------
    if show_structured:
        st.subheader("🧠 Extracted Structured Query")
        st.json(structured_query)

    # ---------------- SORTING ----------------
    if not results.empty:

        if sort_option == "Experience (High to Low)" and "experience" in results.columns:
            results = results.sort_values("experience", ascending=False)

        elif sort_option == "Experience (Low to High)" and "experience" in results.columns:
            results = results.sort_values("experience", ascending=True)

    # ---------------- RESULTS TABLE ----------------
    st.subheader("🏆 Top Coaches")

    if results.empty:
        st.error("No results found.")
    else:

        display_columns = [
            col for col in [
                "username",
                "location",
                "experience",
                "coach_verified",
                "is_alphacoach_assured",
                "certifications_bool",
                "hybrid_score"
            ] if col in results.columns
        ]

        safe_results = results[display_columns].copy()

        # Fix Arrow serialization error
        safe_results = safe_results.fillna("")
        safe_results = safe_results.astype(str)

        st.dataframe(
            safe_results,
            use_container_width=True
        )

    # ---------------- RANKING INFO ----------------
    st.markdown("---")
    st.subheader("📊 Ranking Logic")

    st.markdown("""
**Hybrid Score =**

0.6 × BM25 (Keyword relevance)  
0.4 × Semantic similarity (SBERT)

If quality intent detected (best, top, expert):
- Experience boost  
- Verified boost  
- Assured boost  
- Quality score boost  
""")

st.markdown("---")
st.markdown("Built with LLM + Hybrid Search Architecture 🚀")
