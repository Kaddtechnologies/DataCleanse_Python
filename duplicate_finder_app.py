# -*- coding: utf-8 -*-
"""
Duplicate Finder App – v5.4
===========================
End-to-end Streamlit application for large-scale customer-duplicate detection
**and** reviewer workflow, including:

1. **Dynamic column mapping** – no fixed headers required.
2. **Fast fuzzy matching** – `thefuzz` or `neofuzz` (+ `python-levenshtein`).
3. **Stakeholder views**
   • KPI dashboard • Flash-card reviewer • Interactive grid (st_aggrid).
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
python-dotenv
thefuzz[speedup]     # pulls python-levenshtein
neofuzz              # optional, faster fuzzy
streamlit-aggrid     # interactive grid
xlsxwriter           # Excel export (primary)
openpyxl             # Excel export fallback
------------------------------------------------------------
"""
from __future__ import annotations

import io, json, re, itertools, uuid, os
from pathlib import Path
from typing import List, Dict
import pandas as pd
import streamlit as st
from difflib import ndiff
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# -------------------- ENV + OPENAI ------------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------- FUZZY --------------------------------------------------
from thefuzz import fuzz as _fuzz
def neo_token_set_ratio(a: str, b: str) -> int:
    return _fuzz.token_set_ratio(a, b)

# -------------------- HELPERS ------------------------------------------------
def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text

CANDIDATE_MAP = {
    "customer_name": ["customer", "name", "account", "client"],
    "address"      : ["address", "addr", "street", "road"],
    "city"         : ["city", "town"],
    "country"      : ["country", "nation", "cntry", "co"],
    "tpi"          : ["tpi", "id", "num", "number", "code"],
}

def detect_columns(columns: List[str]) -> Dict[str, str | None]:
    detected = {k: None for k in CANDIDATE_MAP}
    for col in columns:
        col_n = normalize(col)
        for key, hints in CANDIDATE_MAP.items():
            if any(h in col_n for h in hints) and detected[key] is None:
                detected[key] = col
    return detected

# -------------------- DUPLICATE LOGIC ---------------------------------------
def build_duplicate_df(df: pd.DataFrame, col_map: Dict[str, str]) -> pd.DataFrame:
    work = df[[c for c in col_map.values() if c]].copy().reset_index(drop=False).rename(columns={"index":"ExcelRow"})
    for c in ["customer_name","address","city","country"]:
        if col_map[c]:
            work[f"{c}_norm"] = work[col_map[c]].apply(normalize)

    blocks: dict[str, list[int]] = {}
    for i,row in work.iterrows():
        name_prefix = row["customer_name_norm"][:4]
        city_prefix = row["city_norm"][0] if col_map["city"] else ""
        blocks.setdefault(f"{name_prefix}_{city_prefix}", []).append(i)

    st.session_state["block_stats"] = {
        "total_blocks": len(blocks),
        "max_block_size": max(map(len,blocks.values())) if blocks else 0,
        "avg_block_size": (sum(map(len,blocks.values()))/len(blocks)) if blocks else 0,
    }

    master_records: dict[int,dict] = {}
    for idxs in blocks.values():
        for i1,i2 in itertools.combinations(idxs,2):
            r1,r2 = work.loc[i1], work.loc[i2]
            name_s = neo_token_set_ratio(r1[col_map["customer_name"]], r2[col_map["customer_name"]])
            if name_s < 70: continue
            addr_s = neo_token_set_ratio(r1[col_map["address"]], r2[col_map["address"]]) if col_map["address"] else 0
            overall = round((name_s+addr_s)/2)
            if overall<70: continue
            dup = {
                "Row"      : int(r2["ExcelRow"])+2,
                "Name"     : r2[col_map["customer_name"]],
                "Address"  : r2[col_map["address"]] if col_map["address"] else "",
                "Name%"    : name_s,
                "Addr%"    : addr_s,
                "Overall%" : overall,
                "NeedsAI"  : overall<90,
                "LLM_conf" : None,
                "uid"      : str(uuid.uuid4())
            }
            master_row = int(r1["ExcelRow"])+2
            master_records.setdefault(master_row,{
                "Row":master_row,
                "Name":r1[col_map["customer_name"]],
                "Address":r1[col_map["address"]] if col_map["address"] else "",
                "duplicates":[],
                "master_uid":str(uuid.uuid4())
            })["duplicates"].append(dup)

    masters=[]
    for m in master_records.values():
        sims=[d["Overall%"] for d in m["duplicates"]]
        masters.append({
            "MasterRow":m["Row"], "MasterName":m["Name"], "MasterAddress":m["Address"],
            "DuplicateCount":len(m["duplicates"]), "AvgSimilarity": round(sum(sims)/len(sims)) if sims else 0,
            "NeedsAI":any(d["NeedsAI"] for d in m["duplicates"]), "Duplicates":m["duplicates"],
            "master_uid":m["master_uid"]
        })
    return pd.DataFrame(masters).sort_values("AvgSimilarity",ascending=False).reset_index(drop=True)

# -------------------- DIFF ---------------------------------------------------
def diff_html(a:str,b:str)->str:
    return " ".join(
        f"<span style='{ 'background:#fff3cd;text-decoration:line-through;' if t.startswith('- ') else 'background:#d4edda;' if t.startswith('+ ') else '' }padding:2px 4px;border-radius:3px;margin:0 2px;'>{t[2:]}</span>"
        for t in ndiff(a.split(),b.split())
    )

# -------------------- LLM ----------------------------------------------------
def ask_llm_batch(rows:List[dict])->tuple[List[float],str|None]:
    if not rows:return [],None
    rsp=client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role":"system","content":"Return JSON array of probs then 2-sentence summary."},
            {"role":"user","content":json.dumps(rows)}
        ],temperature=0,max_tokens=400
    )
    txt=rsp.choices[0].message.content
    arr=json.loads(txt[txt.find('['):txt.find(']')+1])
    summary=txt[txt.find(']')+1:].strip()
    return [float(x) for x in arr],summary or None

# -------------------- UI -----------------------------------------------------
st.set_page_config("Duplicate Finder","wide")
st.markdown("""
<style>
.stApp{background:#0e1117;color:#fafafa}
.stMarkdown p,.stMarkdown div,.stMarkdown span{color:#fafafa !important}
th,td{color:#fafafa !important}
.stMetric div, .stMetric label{color:#fafafa !important}
.stFileUploader label{color:#fafafa !important}
.stProgress>div>div{background:#4e8df5}
</style>""",unsafe_allow_html=True)

st.title("🔍 Customer Duplicate Finder")

# ---------------- File Upload ------------------------------------------------
st.markdown("### 📤 File Upload")
uploaded=st.file_uploader("Upload Excel or CSV",type=["xlsx","xls","csv"])
if uploaded:
    upl_msg=st.empty();progress=upl_msg.progress(0)
    try:
        if uploaded.name.endswith("csv"):
            import charset_normalizer as cn
            raw=uploaded.read();enc=cn.detect(raw)["encoding"] or "utf-8"
            df_raw=pd.read_csv(io.BytesIO(raw),dtype=str,encoding=enc,na_filter=False)
        else:
            xl=pd.ExcelFile(uploaded);df_raw=xl.parse(xl.sheet_names[0],dtype=str,na_filter=False)
        progress.progress(100,text="")
        upl_msg.success(f"Loaded {len(df_raw):,} rows from {uploaded.name}",icon="✅")
    except Exception as e:
        upl_msg.error(str(e));st.stop()

    # -------------- Column mapping sidebar ---------------------------------
    detected=detect_columns(df_raw.columns.tolist())
    st.sidebar.header("Column Mapping")
    col_map={}
    for k,v in detected.items():
        col_map[k]=st.sidebar.selectbox(k.title().replace("_"," "),[None]+df_raw.columns.tolist(),
                                        index=(1+df_raw.columns.tolist().index(v)) if v else 0)
    st.sidebar.markdown("---")
    run_dedup=st.sidebar.button("Run Deduplication")

    if run_dedup:
        st.session_state["dup_df"]=build_duplicate_df(df_raw,col_map)
        st.success(f"Found {len(st.session_state['dup_df'])} potential duplicates",icon="🗂️")

    dup_df:pd.DataFrame|None=st.session_state.get("dup_df")
    if dup_df is not None and not dup_df.empty:

        # ---------------- KPI Dashboard ------------------------------------
        st.markdown("### 📊 KPI Dashboard")
        hi=dup_df[dup_df["AvgSimilarity"]>=98]
        med=dup_df[(dup_df["AvgSimilarity"]<98)&(dup_df["AvgSimilarity"]>=90)]
        low=dup_df[dup_df["NeedsAI"]]

        c1,c2,c3,c4=st.columns(4)
        c1.metric("Auto-merge",len(hi))
        c2.metric("Needs Review",len(med))
        c3.metric("Needs AI",len(low))
        stats=st.session_state.get("block_stats",{})
        c4.metric("Total Blocks",stats.get("total_blocks",0))
        st.markdown("<div style='font-size:.8em;color:#fafafa66;margin-top:-10px'>Total Blocks: groups of similar records.</div>",
                    unsafe_allow_html=True)
        st.divider()

        # ---------------- Interactive Grid ---------------------------------
        from st_aggrid import AgGrid,GridOptionsBuilder,JsCode
        disp=dup_df.drop(columns=["Duplicates"]).copy()
        disp["View"]="👁️ View"
        jscode=JsCode("""
            function(p){return `<span style='cursor:pointer;background:#4e8df5;color:white;
                                 padding:3px 8px;border-radius:6px;font-weight:600;'>${p.value}</span>`}
        """)
        gb=GridOptionsBuilder.from_dataframe(disp)
        gb.configure_column("View",cellRenderer=jscode,width=110,suppressSizeToFit=True)
        gb.configure_default_column(resizable=True,sortable=True,filter=True)
        gb.configure_selection("multiple",use_checkbox=True,groupSelectsChildren=True)
        grid=AgGrid(disp,gb.build(),height=400)
        selected_rows=grid["selected_rows"]

        # ----------- Dialog on View click ----------------------------------
        if grid.get("clicked_cell") and grid["clicked_cell"]["column"]=="View":
            idx=grid["clicked_cell"]["row"];m=dup_df.iloc[idx]
            with st.dialog(f'Duplicates for "{m["MasterName"]}"',width="large"):
                st.markdown(f"""
                <h3 style='color:#4e8df5;margin:0 0 10px 0'>Master Record (row {m['MasterRow']})</h3>
                <div style='background:#262730;padding:12px;border-radius:6px;margin-bottom:18px'>
                    <b>Name:</b> {m['MasterName']}<br>
                    <b>Address:</b> {m['MasterAddress']}<br>
                    <b>Avg Similarity:</b> {m['AvgSimilarity']}%<br>
                    <b>Potential Duplicates:</b> {m['DuplicateCount']}
                </div>
                """,unsafe_allow_html=True)
                for i,d in enumerate(m["Duplicates"],1):
                    col=lambda v:"#28a745" if v>=90 else "#ffc107" if v>=75 else "#dc3545"
                    st.markdown(f"""
                    <div style='border-left:6px solid {col(d["Overall%"])};background:#1e1e2e;
                                padding:14px;border-radius:6px;margin-bottom:14px'>
                        <h4 style='margin:0 0 6px 0'>Duplicate #{i} (row {d["Row"]})</h4>
                        <div style='display:flex;gap:18px;flex-wrap:wrap'>
                            <div style='flex:1 1 220px'><b>Name</b><br>
                               <span style='color:{col(d["Name%"])};font-weight:600'>{d["Name%"]}%</span> – {d["Name"]}
                            </div>
                            <div style='flex:2 1 320px'><b>Address</b><br>
                               <span style='color:{col(d["Addr%"])};font-weight:600'>{d["Addr%"]}%</span> – {d["Address"]}
                            </div>
                            <div style='flex:0 0 110px;text-align:center'>
                               <b>Overall</b><br>
                               <span style='color:{col(d["Overall%"])};font-size:1.2em;font-weight:700'>
                                   {d["Overall%"]}%
                               </span>
                            </div>
                        </div>
                    </div>
                    """,unsafe_allow_html=True)

        # ---------------- AI Assistance ------------------------------------
        st.subheader("🤖 AI Assistance")
        if st.button("Analyze Selected with AI"):
            targets=[]
            if selected_rows:
                for r in selected_rows:
                    row_idx=r["_selectedRowNodeInfo"]["nodeRowIndex"]
                    targets.extend(dup_df.iloc[row_idx]["Duplicates"])
            else:
                for _,m in dup_df[dup_df["NeedsAI"]].iterrows():
                    targets.extend(m["Duplicates"])
            if not targets:
                st.info("Nothing to analyze.")
            else:
                ai_rows=[]
                for dup in targets:
                    master=next(m for _,m in dup_df.iterrows()
                                if any(d["uid"]==dup["uid"] for d in m["Duplicates"]))
                    ai_rows.append({
                        "Name1":master["MasterName"],"Name2":dup["Name"],
                        "Addr1":master["MasterAddress"],"Addr2":dup["Address"],
                        "Name%":dup["Name%"],"Addr%":dup["Addr%"],
                        "Overall%":dup["Overall%"],"uid":dup["uid"]
                    })
                scores,summary=ask_llm_batch(ai_rows)
                if scores and summary:
                    mapp={r["uid"]:s for r,s in zip(ai_rows,scores)}
                    for i,m in dup_df.iterrows():
                        for j,d in enumerate(m["Duplicates"]):
                            if d["uid"] in mapp:
                                dup_df.at[i,"Duplicates"][j]["LLM_conf"]=mapp[d["uid"]]
                    st.success("AI analysis complete.")
                else:
                    st.error("AI call failed.")

        # ---------------- Flash-card Reviewer ------------------------------
        st.subheader("🔎 Flash-Card Reviewer")
        with st.expander("ℹ️ What is this?",expanded=False):
            st.markdown("""
            Review medium-confidence pairs (90–98 % overall).  
            **L** keep Left **R** keep Right **B** keep Both **S** skip
            """)
        if "card_idx" not in st.session_state: st.session_state["card_idx"]=0
        if "card_decisions" not in st.session_state: st.session_state["card_decisions"]={}
        med_pairs=[]
        for _,m in dup_df.iterrows():
            for d in m["Duplicates"]:
                if 90<=d["Overall%"]<98:
                    med_pairs.append({
                        **{k:m[k] for k in ["MasterRow","MasterName","MasterAddress"]},
                        **{k:d[k] for k in ["Row","Name","Address","Name%","Addr%","Overall%","uid"]}
                    })
        med_df=pd.DataFrame(med_pairs)
        if med_df.empty():
            st.info("No medium-confidence pairs.")
        else:
            idx=st.session_state["card_idx"]%len(med_df);pair=med_df.loc[idx]
            st.write(f"Pair {idx+1}/{len(med_df)} — Overall {pair['Overall%']}%")
            st.markdown(f"""
            **Master** (row {pair['MasterRow']}) vs **Duplicate** (row {pair['Row']})
            """)
            st.markdown(diff_html(pair["MasterName"],pair["Name"]),unsafe_allow_html=True)
            st.markdown(diff_html(pair["MasterAddress"],pair["Address"]),unsafe_allow_html=True)
            colL,colR,colB,colS=st.columns(4)
            if colL.button("Keep Left (L)"):
                st.session_state["card_decisions"][pair["uid"]]="left";st.session_state["card_idx"]+=1;st.experimental_rerun()
            if colR.button("Keep Right (R)"):
                st.session_state["card_decisions"][pair["uid"]]="right";st.session_state["card_idx"]+=1;st.experimental_rerun()
            if colB.button("Keep Both (B)"):
                st.session_state["card_decisions"][pair["uid"]]="both";st.session_state["card_idx"]+=1;st.experimental_rerun()
            if colS.button("Skip (S)"):
                st.session_state["card_idx"]+=1;st.experimental_rerun()

        # ---------------- Export -------------------------------------------
        st.subheader("📤 Export Results")
        def flat(df:pd.DataFrame)->pd.DataFrame:
            rows=[]
            for _,m in df.iterrows():
                rows.append({"RecordType":"Master",**m.drop("Duplicates")})
                for d in m["Duplicates"]:
                    rows.append({"RecordType":"Duplicate",**d,"MasterRow":m["MasterRow"],"MasterName":m["MasterName"]})
            return pd.DataFrame(rows)
        colE,colC=st.columns(2)
        if colE.button("Export Excel"):
            buf=io.BytesIO()
            with pd.ExcelWriter(buf,engine="xlsxwriter") as w:
                flat(dup_df).to_excel(w,index=False)
            st.download_button("Download Excel",buf.getvalue(),
                               file_name=f"duplicates_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
                               mime="application/vnd.ms-excel")
        if colC.button("Export CSV"):
            csv=flat(dup_df).to_csv(index=False)
            st.download_button("Download CSV",csv,
                               file_name=f"duplicates_{datetime.now():%Y%m%d_%H%M%S}.csv",
                               mime="text/csv")
