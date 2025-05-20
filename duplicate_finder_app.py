# -*- coding: utf-8 -*-
"""
Duplicate Finder App – v5.2
==========================
End‑to‑end Streamlit application for large‑scale customer‑duplicate detection
**and** reviewer workflow, including:

1. **Dynamic column mapping** – no fixed headers required.
2. **Fast fuzzy matching** – `thefuzz` or `neofuzz` (+ `python‑levenshtein`).
3. **Stakeholder views**
   • KPI dashboard • Flash‑card reviewer • Interactive grid.
4. **Inline diff citations** and Excel/CSV export.
5. **LLM batch assist**
   • Rows with `Overall% < 90` flagged "🤖 Needs AI".
   • Users can select *all / some / one* rows then click **Analyze Selected with AI**.
   • Placeholder `ask_llm_batch()` ready for OpenAI call.

------------------------------------------------------------
Requirements (add to requirements.txt)
------------------------------------------------------------
streamlit>=1.33
pandas
charset-normalizer
python-dateutil
pathlib

thefuzz[speedup]     # pulls python-levenshtein
neofuzz              # optional, faster fuzzy
st_aggrid            # optional interactive grid
xlsxwriter           # Excel export (primary)
openpyxl             # Excel export fallback

------------------------------------------------------------
"""
from __future__ import annotations

import io, json, re, random, itertools, uuid
from pathlib import Path
from typing import List, Dict
import os
import pandas as pd
import streamlit as st
from difflib import ndiff
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Fuzzy matching ---------------------------------------------------------
# Always use thefuzz for token_set_ratio to avoid neofuzz attribute error
from thefuzz import fuzz as _fuzz

def neo_token_set_ratio(a: str, b: str) -> int:
    return _fuzz.token_set_ratio(a, b)

# --- Helpers ----------------------------------------------------------------

def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

# Column detection keywords
CANDIDATE_MAP = {
    "customer_name": ["customer", "name", "account", "client"],
    "address": ["address", "addr", "street", "road"],
    "city": ["city", "town"],
    "country": ["country", "nation", "cntry", "co"],
    "tpi": ["tpi", "id", "num", "number", "code"],
}


def detect_columns(columns: List[str]) -> Dict[str, str | None]:
    detected = {k: None for k in CANDIDATE_MAP}
    for col in columns:
        col_n = normalize(col)
        for key, hints in CANDIDATE_MAP.items():
            if any(h in col_n for h in hints):
                if detected[key] is None:
                    detected[key] = col
    return detected


# ---------------------------------------------------------------------------
# Duplicate detection logic
# ---------------------------------------------------------------------------
NAME_TH = 90
ADDR_TH = 85
ACC_TH = 90  # if global account included


def build_duplicate_df(df: pd.DataFrame, col_map: Dict[str, str]) -> pd.DataFrame:
    cols = col_map
    work = df[[cols[c] for c in cols if cols[c] is not None]].copy()
    work = work.reset_index(drop=False).rename(columns={"index": "ExcelRow"})

    # normalize comparable strings
    for c in ["customer_name", "address", "city", "country"]:
        if cols[c]:
            work[f"{c}_norm"] = work[cols[c]].apply(normalize)

    # More effective blocking strategy
    blocks = {}
    for idx, row in work.iterrows():
        # Use first 4 letters of name + first letter of city if available
        name_prefix = row["customer_name_norm"][:4] if len(row["customer_name_norm"]) >= 4 else row["customer_name_norm"]
        city_prefix = ""
        if cols["city"] and len(row[f"city_norm"]) > 0:
            city_prefix = row[f"city_norm"][0]
        
        key = f"{name_prefix}_{city_prefix}"
        blocks.setdefault(key, []).append(idx)
    
    # Log blocking statistics
    block_sizes = [len(block) for block in blocks.values()]
    st.session_state["block_stats"] = {
        "total_blocks": len(blocks),
        "max_block_size": max(block_sizes) if block_sizes else 0,
        "avg_block_size": sum(block_sizes)/len(block_sizes) if block_sizes else 0
    }
    
    # Set a reasonable limit for comparisons per block
    MAX_COMPARISONS_PER_BLOCK = 1000
    MAX_TOTAL_COMPARISONS = 10000
    
    pairs = []
    total_comparisons = 0
    
    for block_key, idxs in blocks.items():
        if len(idxs) < 2:
            continue
            
        # If block is too large, sample or limit comparisons
        block_comparisons = []
        if len(idxs) > 100:  # Large block
            # Sort by name to compare similar names first
            sorted_idxs = sorted(idxs, key=lambda i: work.loc[i, "customer_name_norm"])
            # Take combinations of nearby records
            for i in range(len(sorted_idxs)-1):
                for j in range(i+1, min(i+20, len(sorted_idxs))):
                    block_comparisons.append((sorted_idxs[i], sorted_idxs[j]))
                    if len(block_comparisons) >= MAX_COMPARISONS_PER_BLOCK:
                        break
                if len(block_comparisons) >= MAX_COMPARISONS_PER_BLOCK:
                    break
        else:
            # For smaller blocks, take all combinations
            block_comparisons = list(itertools.combinations(idxs, 2))[:MAX_COMPARISONS_PER_BLOCK]
        
        # Process the comparisons for this block
        for i1, i2 in block_comparisons:
            r1, r2 = work.loc[i1], work.loc[i2]
            name_s = neo_token_set_ratio(r1[cols["customer_name"]], r2[cols["customer_name"]])
            
            # Skip low name similarity early
            if name_s < 70:
                continue
                
            addr_s = neo_token_set_ratio(r1[cols["address"]], r2[cols["address"]]) if cols["address"] else 0
            overall = round((name_s + addr_s) / 2)
            
            # Only include pairs with reasonable similarity
            if overall >= 70:
                needs_ai = overall < 90
                pairs.append({
                "Row1": int(r1["ExcelRow"])+2,
                "Row2": int(r2["ExcelRow"])+2,
                "Name1": r1[cols["customer_name"]],
                "Name2": r2[cols["customer_name"]],
                "Addr1": r1[cols["address"]] if cols["address"] else "",
                "Addr2": r2[cols["address"]] if cols["address"] else "",
                "Name%": name_s,
                "Addr%": addr_s,
                "Overall%": overall,
                "NeedsAI": needs_ai,
                "LLM_conf": None,
                "uid": str(uuid.uuid4())
            })
    dup_df = pd.DataFrame(pairs)
    return dup_df.sort_values("Overall%", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Diff HTML for snippet viewer
# ---------------------------------------------------------------------------

def diff_html(a: str, b: str) -> str:
    """Generate HTML that highlights differences between two strings with improved readability."""
    tokens = ndiff(a.split(), b.split())
    out = []
    for t in tokens:
        if t.startswith("- "):
            out.append(f"<span style='background:#fff3cd;text-decoration:line-through;padding:2px 4px;margin:0 2px;border-radius:3px;font-weight:500;color:#856404'>{t[2:]}</span>")
        elif t.startswith("+ "):
            out.append(f"<span style='background:#d4edda;padding:2px 4px;margin:0 2px;border-radius:3px;font-weight:500;color:#155724'>{t[2:]}</span>")
        else:
            out.append(f"<span style='margin:0 2px'>{t[2:]}</span>")
    return " ".join(out)


# ---------------------------------------------------------------------------
# Placeholder AI helpers
# ---------------------------------------------------------------------------

def ask_llm_batch(rows: List[dict]) -> List[float]:
    """
    Sends a batch of duplicate-pair dicts to OpenAI and
    returns a list of confidence scores (floats 0-1).

    • If the call fails, None is returned for each row so the UI can handle it.
    """
    if not rows:
        return []

    # ---- Build prompt ----------------------------------------------------
    system_msg = (
        "You are a data-quality assistant. For each pair of customer records "
        "in the JSON list provided by the user, respond ONLY with a JSON array "
        "of floating-point numbers (0-1) representing the probability that the "
        "pair refers to the same underlying customer. Maintain the same order."
    )
    user_msg = json.dumps(rows, indent=2)

    try:
        # Using the chat completions API with the OpenAI client
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        
        # Extract content from the response
        content = response.choices[0].message.content
        scores = json.loads(content)
        return [float(s) for s in scores]

    except Exception as e:
        st.error(f"LLM batch call failed: {e}")
        return [None] * len(rows)



def auto_supersede(df_pairs: pd.DataFrame):
    # Placeholder for rule/ML merge output
    pass


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

# Set dark mode and wide layout
st.set_page_config(
    page_title="Duplicate Finder",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "Customer Duplicate Finder v5.2"
    }
)

# Enhanced dark theme with improved styling for flash cards and overall UI
st.markdown("""
<style>
    /* Base app styling */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #262730;
        color: #fafafa;
        border: 1px solid #4e4e4e;
        border-radius: 5px;
        transition: all 0.2s ease;
    }
    
    .stButton button:hover {
        background-color: #3a3b47;
        border-color: #6e6e6e;
        transform: translateY(-1px);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #4e8df5;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 1.1em;
        font-weight: 600;
        color: #4e8df5;
        background-color: #1e1e2e;
        border-radius: 5px;
        padding: 10px !important;
    }
    
    /* Flash card specific styling */
    .flash-card {
        background-color: #1e1e2e;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .flash-card:hover {
        transform: translateY(-2px);
    }
    
    /* Improve text readability */
    h1, h2, h3, h4 {
        font-weight: 600;
        margin-bottom: 0.5em;
    }
    
    /* Improve table styling */
    table {
        width: 100%;
        border-collapse: collapse;
    }
    
    th, td {
        padding: 8px 12px;
        text-align: left;
        border-bottom: 1px solid #4e4e4e;
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #4e8df5;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #262730;
        color: #fafafa;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Decision buttons */
    .decision-btn {
        margin-bottom: 10px;
        font-weight: 500;
    }
    
    /* Navigation controls */
    .nav-controls {
        display: flex;
        justify-content: space-between;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)
st.title("🔍 Customer Duplicate Finder")

# File Upload Section Instructions
st.markdown("""
### 📤 File Upload
**What it does:** Upload your customer data file in Excel or CSV format to begin the duplicate detection process.
**Why use it:** This is the starting point for finding duplicate customer records in your dataset.
""")

uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xls", "csv"])

if uploaded:
    # Create a placeholder for the progress bar
    upload_progress_placeholder = st.empty()
    progress_bar = upload_progress_placeholder.progress(0, text="Starting file upload process...")
    
    try:
        # File loading process with detailed progress updates
        if uploaded.name.endswith("csv"):
            # CSV processing
            progress_bar.progress(10, text="Reading CSV file...")
            import charset_normalizer as cn
            raw = uploaded.read()
            
            progress_bar.progress(30, text="Detecting file encoding...")
            enc = cn.detect(raw)["encoding"] or "utf-8"
            
            progress_bar.progress(50, text=f"Parsing CSV with {enc} encoding...")
            df_raw = pd.read_csv(io.BytesIO(raw), dtype=str, encoding=enc, na_filter=False)
            
            progress_bar.progress(90, text="Finalizing data import...")
        else:
            # Excel processing
            progress_bar.progress(10, text="Reading Excel file...")
            xl = pd.ExcelFile(uploaded)
            
            progress_bar.progress(40, text=f"Parsing sheet: {xl.sheet_names[0]}...")
            df_raw = xl.parse(xl.sheet_names[0], dtype=str, na_filter=False)
            
            progress_bar.progress(90, text="Finalizing data import...")
        
        # Complete the progress
        progress_bar.progress(100, text="File loaded successfully!")
        
        # Replace progress bar with success message
        upload_progress_placeholder.success(f"Loaded {len(df_raw):,} rows from {uploaded.name}")
    except Exception as e:
        # Show error if file loading fails
        upload_progress_placeholder.error(f"Error loading file: {str(e)}")
        st.stop()

    # Column mapping with instructions
    detected = detect_columns(list(df_raw.columns))
    st.sidebar.header("Column Mapping")
    
    st.sidebar.markdown("""
    **What it does:** Maps your file's columns to standard fields needed for duplicate detection.
    **Why use it:** Allows the system to work with any file format by identifying which columns contain customer names, addresses, etc.
    """)
    
    col_map = {}
    for key, col in detected.items():
        col_map[key] = st.sidebar.selectbox(
            key.title().replace("_", " "), options=[None] + list(df_raw.columns), index=(1 + list(df_raw.columns).index(col)) if col else 0
        )

    # Deduplication section with instructions
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **What it does:** Analyzes your data to find potential duplicate customer records.
    **Why use it:** Identifies similar customer records that might represent the same entity, helping clean your database.
    """)
    run_dedup = st.sidebar.button("Run Deduplication", use_container_width=True)
    
    # Create a placeholder for the deduplication progress
    dedup_progress_placeholder = st.sidebar.empty()
    
    # Create a persistent results display area
    results_placeholder = st.sidebar.empty()
    
    # Show persistent results if available
    if "last_dedup_result" in st.session_state and not run_dedup:
        results_placeholder.success(st.session_state["last_dedup_result"])
    
    if run_dedup:
        try:
            # Detailed progress for deduplication process
            progress_bar = dedup_progress_placeholder.progress(0, text="Starting deduplication...")
            
            # Simulate progress for different stages of deduplication
            progress_bar.progress(10, text="Preparing data for matching...")
            
            # Normalize data
            progress_bar.progress(30, text="Normalizing customer data...")
            
            # Blocking step
            progress_bar.progress(50, text="Creating matching blocks...")
            
            # Pair comparison
            progress_bar.progress(70, text="Comparing potential duplicates...")
            
            # Build the duplicate dataframe
            dup_df = build_duplicate_df(df_raw, col_map)
            
            # Final processing
            progress_bar.progress(90, text="Finalizing results...")
            
            # Store in session state
            st.session_state["dup_df"] = dup_df
            
            # Complete the progress
            progress_bar.progress(100, text="Deduplication complete!")
            
            # Create and store a persistent success message
            if len(dup_df) > 0:
                result_message = f"Found {len(dup_df)} potential duplicates"
                dedup_progress_placeholder.success(result_message)
                st.session_state["last_dedup_result"] = result_message
            else:
                result_message = "No duplicates found"
                dedup_progress_placeholder.info(result_message)
                st.session_state["last_dedup_result"] = result_message
        except Exception as e:
            # Show error if deduplication fails
            dedup_progress_placeholder.error(f"Deduplication failed: {str(e)}")

    dup_df: pd.DataFrame | None = st.session_state.get("dup_df")
    if dup_df is not None and not dup_df.empty:
        # KPI Dashboard with instructions
        st.markdown("""
        ### 📊 KPI Dashboard
        **What it does:** Provides a summary of duplicate detection results categorized by confidence level.
        **Why use it:** Gives you a quick overview of how many duplicates were found and what action they need.
        """)
        
        hi = dup_df[dup_df["Overall%"] >= 98]
        med = dup_df[(dup_df["Overall%"] < 98) & (dup_df["Overall%"] >= 90)]
        low = dup_df[dup_df["Overall%"] < 90]
        
        # Create 4 columns for metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Auto‑merge", len(hi))
        c2.metric("Needs Review", len(med))
        c3.metric("Needs AI", len(low))
        
        # Show blocking statistics with tooltip explanation
        if "block_stats" in st.session_state:
            stats = st.session_state["block_stats"]
            c4.metric("Total Blocks", stats["total_blocks"])
            st.markdown("""
            <div style="font-size:0.8em; color:#aaa; margin-top:-15px;">
            <b>Total Blocks</b>: Groups of similar records that were compared to find duplicates.
            More blocks means more efficient processing with fewer unnecessary comparisons.
            </div>
            """, unsafe_allow_html=True)
            
        st.divider()

        # Interactive Grid with instructions
        st.markdown("""
        ### 📋 Interactive Results Grid
        **What it does:** Displays all potential duplicate pairs in a sortable, filterable table.
        **Why use it:** Allows you to explore all detected duplicates, sort by confidence level, and select rows for AI analysis.
        """)
        
        # Interactive grid (basic DataFrame if st_aggrid not installed)
        try:
            from st_aggrid import AgGrid, GridOptionsBuilder

            gb = GridOptionsBuilder.from_dataframe(dup_df)
            # Set reasonable page size and pagination options
            gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=25)
            gb.configure_default_column(resizable=True, filter=True, sortable=True)
            gb.configure_column("NeedsAI", cellRenderer="function(params){return params.value ? '🤖' : ''}")
            # Add checkbox selection column as first column
            gb.configure_selection("multiple", use_checkbox=True, groupSelectsChildren=True, groupSelectsFiltered=True)
            gb.configure_grid_options(suppressRowClickSelection=True)
            
            # Add pagination status panel
            gb.configure_grid_options(
                paginationAutoPageSize=False,
                pagination=True,
                domLayout='normal',
                rowBuffer=25
            )
            
            grid_res = AgGrid(dup_df, gb.build(), height=400, key="grid")
            selected = grid_res["selected_rows"]
        except ModuleNotFoundError:
            st.info("Install st_aggrid for richer table interactivity.")
            selected = []
            st.dataframe(dup_df)

        # AI Assistance with instructions
        st.subheader("🤖 AI Assistance")
        st.markdown("""
        **What it does:** Uses AI to analyze potential duplicates with lower confidence scores.
        **Why use it:** Helps determine if records are truly duplicates when fuzzy matching alone isn't conclusive.
        
        Select specific rows from the grid above (or leave blank to process all 'Needs AI' rows) and click the button below.
        """)
        
        # Create columns for button and status indicator
        ai_col1, ai_col2 = st.columns([1, 3])
        
        with ai_col1:
            analyze_button = st.button("Analyze Selected with AI")
        
        with ai_col2:
            status_placeholder = st.empty()
            
        if analyze_button:
            # Fix the DataFrame truth value error by checking if selected is empty
            if isinstance(selected, list) and len(selected) > 0:
                target_rows = selected
            else:
                target_rows = dup_df[dup_df["NeedsAI"]].to_dict("records")
                
            if not target_rows or len(target_rows) == 0:
                status_placeholder.info("No rows require AI analysis.")
            else:
                progress_bar = status_placeholder.progress(0)
                progress_bar.progress(10, text="Preparing data for AI analysis...")

            try:
                progress_bar.progress(30, text=f"Sending {len(target_rows)} rows to AI...")
                scores = ask_llm_batch(target_rows)

                progress_bar.progress(70, text="Processing AI results...")
                score_map = {r["uid"]: s for r, s in zip(target_rows, scores)}
                dup_df["LLM_conf"] = dup_df.apply(lambda r: score_map.get(r["uid"], r["LLM_conf"]), axis=1)

                progress_bar.progress(100, text="AI analysis complete!")
                st.session_state["dup_df"] = dup_df

                status_placeholder.success("AI analysis complete. Column 'LLM_conf' updated.")
            except Exception as e:
                status_placeholder.error(f"AI analysis failed: {str(e)}")


        # Flash Card Reviewer
        st.subheader("🔎 Flash Card Reviewer")
        
        # Help and instructions
        with st.expander("ℹ️ What is this and how to use it?", expanded=True):
            st.markdown("""
            ### About Flash Card Reviewer
            
            This tool helps you quickly review and decide on potential duplicate customer records that have been identified with medium confidence (90-98% match).
            
            #### How to use:
            1. **Review the differences** - Highlighted text shows what's different between the two records
                * <span style='background:#fff3cd;text-decoration:line-through;padding:2px 4px;border-radius:3px;color:#856404'>Yellow strikethrough</span> shows text only in the left record
                * <span style='background:#d4edda;padding:2px 4px;border-radius:3px;color:#155724'>Green highlight</span> shows text only in the right record
            
            2. **Make a decision** using one of the buttons:
                * **Keep Left** - The left record is correct or preferred
                * **Keep Right** - The right record is correct or preferred
                * **Keep Both** - These are actually different customers (not duplicates)
                * **Skip** - Come back to this pair later
                
            3. **Keyboard shortcuts** are available:
                * **L** - Keep Left
                * **R** - Keep Right
                * **B** - Keep Both
                * **S** - Skip to next
                * **←** - Previous pair
                * **→** - Next pair
            """, unsafe_allow_html=True)
        
        # Initialize session state for card index if not exists
        if "card_idx" not in st.session_state:
            st.session_state["card_idx"] = 0
            
        # Initialize decisions dictionary if not exists
        if "card_decisions" not in st.session_state:
            st.session_state["card_decisions"] = {}

        # Get medium confidence pairs
        med_df = dup_df[(dup_df["Overall%"] < 98) & (dup_df["Overall%"] >= 90)].reset_index(drop=True)
        
        # Display progress metrics
        if not med_df.empty:
            total_pairs = len(med_df)
            decisions_made = len(st.session_state.get("card_decisions", {}))
            progress_pct = int((decisions_made / total_pairs) * 100) if total_pairs > 0 else 0
            
            # Progress metrics
            st.progress(progress_pct/100, f"Progress: {progress_pct}% ({decisions_made}/{total_pairs} pairs reviewed)")
            
            # Card navigation
            idx = st.session_state["card_idx"] % len(med_df)
            row = med_df.loc[idx]
            pair_id = row["uid"]
            
            # Card container with styling
            with st.container():
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; border: 1px solid #4e4e4e;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;
                            background-color: #1e1e1e;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                        <h3 style="margin: 0;">Pair {idx+1} of {len(med_df)}</h3>
                        <span style="background-color: #4e8df5; color: white; padding: 3px 10px;
                                border-radius: 20px; font-weight: bold;">
                            {row['Overall%']}% Match
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Create two columns for comparison
                col1, col2 = st.columns(2)
                
                # Left column - Record details
                with col1:
                    st.markdown("<div style='background-color:#262730; padding:15px; border-radius:5px; margin-bottom:15px;'>", unsafe_allow_html=True)
                    st.markdown("<h4>Customer Name</h4>", unsafe_allow_html=True)
                    st.markdown(diff_html(row["Name1"], row["Name2"]), unsafe_allow_html=True)
                    
                    st.markdown("<h4 style='margin-top:15px;'>Address</h4>", unsafe_allow_html=True)
                    st.markdown(diff_html(row["Addr1"], row["Addr2"]), unsafe_allow_html=True)
                    
                    # Add more details if available
                    st.markdown("<h4 style='margin-top:15px;'>Record Details</h4>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <table style="width:100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding:5px; border-bottom:1px solid #4e4e4e; width:50%;">Left Record (Row {row['Row1']})</td>
                            <td style="padding:5px; border-bottom:1px solid #4e4e4e; width:50%;">Right Record (Row {row['Row2']})</td>
                        </tr>
                        <tr>
                            <td style="padding:5px; color:#aaa;">Name Match: {row['Name%']}%</td>
                            <td style="padding:5px; color:#aaa;">Address Match: {row['Addr%']}%</td>
                        </tr>
                    </table>
                    """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Right column - Decision buttons
                with col2:
                    st.markdown("<div style='background-color:#262730; padding:15px; border-radius:5px;'>", unsafe_allow_html=True)
                    st.markdown("<h4>Make a Decision</h4>", unsafe_allow_html=True)
                    
                    # Decision buttons with icons and better styling
                    if st.button("👈 Keep Left (L)", key="keep_left", use_container_width=True):
                        st.session_state["card_decisions"][pair_id] = "left"
                        st.session_state["card_idx"] += 1
                        st.experimental_rerun()
                        
                    if st.button("👉 Keep Right (R)", key="keep_right", use_container_width=True):
                        st.session_state["card_decisions"][pair_id] = "right"
                        st.session_state["card_idx"] += 1
                        st.experimental_rerun()
                        
                    if st.button("🔄 Keep Both (B)", key="keep_both", use_container_width=True):
                        st.session_state["card_decisions"][pair_id] = "both"
                        st.session_state["card_idx"] += 1
                        st.experimental_rerun()
                    
                    st.markdown("<hr style='margin:15px 0;'>", unsafe_allow_html=True)
                    
                    # Navigation buttons
                    nav_col1, nav_col2, nav_col3 = st.columns([1,2,1])
                    with nav_col1:
                        if st.button("⬅️ Prev", key="prev_card"):
                            st.session_state["card_idx"] -= 1
                            st.experimental_rerun()
                    
                    with nav_col2:
                        if st.button("⏭️ Skip (S)", key="skip_card", use_container_width=True):
                            st.session_state["card_idx"] += 1
                            st.experimental_rerun()
                    
                    with nav_col3:
                        if st.button("➡️ Next", key="next_card"):
                            st.session_state["card_idx"] += 1
                            st.experimental_rerun()
                    
                    # Keyboard shortcuts
                    st.markdown("""
                    <script>
                    document.addEventListener('keydown', function(e) {
                        if (e.key === 'l' || e.key === 'L') {
                            document.querySelector('button[key="keep_left"]').click();
                        } else if (e.key === 'r' || e.key === 'R') {
                            document.querySelector('button[key="keep_right"]').click();
                        } else if (e.key === 'b' || e.key === 'B') {
                            document.querySelector('button[key="keep_both"]').click();
                        } else if (e.key === 's' || e.key === 'S') {
                            document.querySelector('button[key="skip_card"]').click();
                        } else if (e.key === 'ArrowLeft') {
                            document.querySelector('button[key="prev_card"]').click();
                        } else if (e.key === 'ArrowRight') {
                            document.querySelector('button[key="next_card"]').click();
                        }
                    });
                    </script>
                    """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No medium-confidence pairs to review.")

        # Export options
        st.subheader("📊 Export Results")
        st.markdown("""
        **What it does:** Exports your duplicate detection results to Excel or CSV format.
        **Why use it:** Allows you to save, share, or further process the results in other systems.
        """)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export to Excel"):
                try:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                        dup_df.to_excel(writer, sheet_name="Duplicates", index=False)
                    st.download_button(
                        label="Download Excel",
                        data=buffer.getvalue(),
                        file_name=f"duplicates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                except Exception as e:
                    st.error(f"Excel export failed: {e}")
                    try:
                        # Fallback to openpyxl
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                            dup_df.to_excel(writer, sheet_name="Duplicates", index=False)
                        st.download_button(
                            label="Download Excel (Fallback)",
                            data=buffer.getvalue(),
                            file_name=f"duplicates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.ms-excel"
                        )
                    except Exception as e2:
                        st.error(f"Fallback export also failed: {e2}")
        with col2:
            if st.button("Export to CSV"):
                try:
                    csv = dup_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"duplicates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"CSV export failed: {e}")