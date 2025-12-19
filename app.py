import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import os
import google.generativeai as genai

st.set_page_config(page_title="HDB AI Analyst", layout="wide")

# --- 1. SECURE API KEY HANDLING ---
# Try to get the key from Streamlit Secrets
try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        # Fallback for local testing if secrets.toml is missing (Not recommended for prod)
        api_key = os.getenv("GEMINI_API_KEY") 
except FileNotFoundError:
    api_key = None

# --- CONFIGURATION ---
RESOURCE_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
CACHE_FILE = "hdb_full_data_cache.csv"
TOWN_COORDS = {
    "ANG MO KIO": {"lat": 1.3691, "lon": 103.8454},
    "BEDOK": {"lat": 1.3236, "lon": 103.9273},
    "BISHAN": {"lat": 1.3526, "lon": 103.8352},
    "BUKIT BATOK": {"lat": 1.3590, "lon": 103.7637},
    "BUKIT MERAH": {"lat": 1.2735, "lon": 103.8232},
    "BUKIT PANJANG": {"lat": 1.3774, "lon": 103.7719},
    "BUKIT TIMAH": {"lat": 1.3294, "lon": 103.8155},
    "CENTRAL AREA": {"lat": 1.2789, "lon": 103.8536},
    "CHOA CHU KANG": {"lat": 1.3840, "lon": 103.7470},
    "CLEMENTI": {"lat": 1.3140, "lon": 103.7624},
    "GEYLANG": {"lat": 1.3201, "lon": 103.8918},
    "HOUGANG": {"lat": 1.3612, "lon": 103.8863},
    "JURONG EAST": {"lat": 1.3329, "lon": 103.7436},
    "JURONG WEST": {"lat": 1.3404, "lon": 103.7090},
    "KALLANG/WHAMPOA": {"lat": 1.3100, "lon": 103.8651},
    "LIM CHU KANG": {"lat": 1.4342, "lon": 103.7149},
    "MARINE PARADE": {"lat": 1.3020, "lon": 103.9073},
    "PASIR RIS": {"lat": 1.3721, "lon": 103.9474},
    "PUNGGOL": {"lat": 1.3984, "lon": 103.9072},
    "QUEENSTOWN": {"lat": 1.2942, "lon": 103.8061},
    "SEMBAWANG": {"lat": 1.4491, "lon": 103.8185},
    "SENGKANG": {"lat": 1.3868, "lon": 103.8914},
    "SERANGOON": {"lat": 1.3554, "lon": 103.8713},
    "TAMPINES": {"lat": 1.3555, "lon": 103.9431},
    "TOA PAYOH": {"lat": 1.3343, "lon": 103.8563},
    "WOODLANDS": {"lat": 1.4382, "lon": 103.7890},
    "YISHUN": {"lat": 1.4304, "lon": 103.8354}
}

# --- AI ENGINE ---
def ask_ai(question, df):
    if not api_key:
        return "⚠️ API Key missing. Please set GEMINI_API_KEY in secrets.", None
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    columns = list(df.columns)
    sample_data = df.head(3).to_string()
    
    prompt = f"""
    You are a Python Data Analyst. 
    You have a pandas dataframe named 'df'.
    Columns: {columns}
    Sample: {sample_data}
    
    User Query: "{question}"
    
    Task: Write 1 line of Python code to answer. Return ONLY the code. No markdown.
    Example: df[df['town']=='BEDOK']['resale_price'].mean()
    """
    
    try:
        response = model.generate_content(prompt)
        generated_code = response.text.strip().replace("```python", "").replace("```", "")
        local_vars = {"df": df, "pd": pd}
        result = eval(generated_code, {"__builtins__": None}, local_vars)
        return result, generated_code
    except Exception as e:
        return f"Could not calculate. ({e})", None

# --- DATA LOADER ---
@st.cache_data(ttl=3600)
def load_all_data():
    if os.path.exists(CACHE_FILE):
        return pd.read_csv(CACHE_FILE)

    base_url = "https://data.gov.sg/api/action/datastore_search"
    all_records = []
    offset = 0
    limit = 10000 
    
    progress = st.progress(0)
    status = st.empty()
    
    try:
        while True:
            params = {"resource_id": RESOURCE_ID, "limit": limit, "offset": offset, "sort": "month desc"}
            r = requests.get(base_url, params=params)
            data = r.json()
            if not data.get('success') or not data['result']['records']: break
            all_records.extend(data['result']['records'])
            status.text(f"Downloaded {len(all_records):,} records...")
            if len(data['result']['records']) < limit: break
            offset += limit
            if len(all_records) > 300000: break

        progress.empty()
        status.empty()
        df = pd.DataFrame(all_records)
        if 'month' in df.columns: df['month'] = pd.to_datetime(df['month'])
        for c in ['resale_price', 'floor_area_sqm', 'lease_commence_date']:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
        df.to_csv(CACHE_FILE, index=False)
        return df
    except Exception: return pd.DataFrame()

# --- UI LAYOUT ---
st.title("🇸🇬 HDB AI Analyst")
df = load_all_data()

if not df.empty and 'month' in df.columns:
    df['month'] = pd.to_datetime(df['month'])

if not df.empty:
    # --- AI CHATBOT ---
    st.markdown("### 🤖 Ask the AI")
    
    if not api_key:
        st.warning("🔴 **Developer Note:** API Key is not configured. Please add `GEMINI_API_KEY` to your secrets.")
    else:
        question = st.text_input("Ask a question:", placeholder="e.g., 'What is the highest price in Bishan?'")
        if question:
            with st.spinner("Thinking..."):
                answer, code = ask_ai(question, df)
                st.success(f"**Answer:** {answer}")
                if code:
                    with st.expander("Show Logic"):
                        st.code(code, language="python")

    st.divider()

    # --- STANDARD DASHBOARD ---
    st.subheader("📊 Visual Explorer")
    c1, c2 = st.columns(2)
    with c1:
        towns = sorted(df['town'].astype(str).unique())
        sel_towns = st.multiselect("Towns", towns, default=["ANG MO KIO", "BEDOK"])
    with c2:
        types = sorted(df['flat_type'].astype(str).unique())
        sel_types = st.mult