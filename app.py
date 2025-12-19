import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import os
import google.generativeai as genai
import re

st.set_page_config(page_title="HDB AI Analyst", layout="wide")

# --- 1. SECURE API KEY ---
try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
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
    
    try:
        while True:
            params = {"resource_id": RESOURCE_ID, "limit": limit, "offset": offset, "sort": "month desc"}
            r = requests.get(base_url, params=params)
            data = r.json()
            if not data.get('success') or not data['result']['records']: break
            all_records.extend(data['result']['records'])
            if len(data['result']['records']) < limit: break
            offset += limit
            if len(all_records) > 300000: break

        progress.empty()
        df = pd.DataFrame(all_records)
        if 'month' in df.columns: df['month'] = pd.to_datetime(df['month'])
        for c in ['resale_price', 'floor_area_sqm', 'lease_commence_date']:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
        df.to_csv(CACHE_FILE, index=False)
        return df
    except Exception: return pd.DataFrame()

# --- HELPER: CONTEXT GENERATOR ---
def get_dataset_context(df):
    """Summarizes data (Price & Volume) for the AI."""
    # 1. Setup Data
    df_context = df.copy()
    df_context['year'] = df_context['month'].dt.year
    latest_year = df_context['year'].max()
    
    # 2. Price Map (Recent)
    recent_df = df_context[df_context['year'] == latest_year]
    price_map = recent_df.groupby(['town', 'flat_type'])['resale_price'].mean().to_dict()
    
    # 3. Yearly Trends (Price AND Volume)
    yearly_price = df_context.groupby('year')['resale_price'].mean().to_dict()
    yearly_vol = df_context.groupby('year')['resale_price'].count().to_dict()
    
    # 4. Peak Volume Month
    monthly_counts = df_context.groupby('month').size()
    peak_month = monthly_counts.idxmax().strftime('%Y-%m')
    peak_val = monthly_counts.max()
    
    return f"""
    [Current Year ({latest_year}) Prices]: {price_map}
    [Yearly Avg Price]: {yearly_price}
    [Yearly Volume (Transactions)]: {yearly_vol}
    [Highest Volume Month All-Time]: {peak_month} ({peak_val} sales)
    """

# --- AI ENGINE ---
def ask_ai(question, df):
    if not api_key:
        return "⚠️ API Key missing.", None
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    context_str = get_dataset_context(df)
    
    prompt = f"""
    You are a Singapore Property Consultant.
    
    DATA CONTEXT:
    {context_str}
    
    User Question: "{question}"
    
    STRICT RULES:
    1. **FUTURE PREDICTIONS** (e.g. "Price in 2026?"): Do NOT write code. Write TEXT estimate based on trend.
    2. **ADVICE** (e.g. "Should I sell?"): Write TEXT only.
    3. **CALCULATIONS** (e.g. "Most expensive town?", "Average price?", "Highest transactions?"): 
       - Write Python code wrapped in ```python ... ```.
       - IMPORTANT: Assign your final answer (as a string) to a variable named `result`.
       - Example:
         ```python
         top = df.groupby('town')['resale_price'].mean().idxmax()
         result = f"The most expensive town is {{top}}."
         ```
    """
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Check for code block
        code_match = re.search(r"```python(.*?)```", text, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            
            if "import" in code or "open(" in code: 
                return "I cannot execute that command for safety reasons.", None
            
            # Execution Environment
            local_vars = {"df": df, "pd": pd, "result": None}
            
            try:
                # Use standard globals (None) to allow len(), max(), etc.
                exec(code, {}, local_vars)
                return local_vars.get("result", "Calculation finished but no result returned."), code
            except Exception as e:
                return f"Calculation Error: {e}", code
        else:
            return text, None
            
    except Exception as e:
        return f"AI Error: {e}", None

# --- MAIN APP UI ---
st.title("🇸🇬 HDB AI Analyst")

df = load_all_data()
if not df.empty and 'month' in df.columns:
    df['month'] = pd.to_datetime(df['month'])

if not df.empty:
    
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filter Settings")
    
    # 1. Date Filter
    min_date = df['month'].min().date()
    max_date = df['month'].max().date()
    start_date, end_date = st.sidebar.date_input(
        "Date Range", 
        value=(min_date, max_date), 
        min_value=min_date, 
        max_value=max_date
    )

    # 2. Town Filter
    all_towns = sorted(df['town'].astype(str).unique())
    sel_towns = st.sidebar.multiselect("Towns", all_towns, default=["ANG MO KIO", "BEDOK"])
    
    # 3. Type Filter
    all_types = sorted