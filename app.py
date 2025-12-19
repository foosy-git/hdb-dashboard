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

# --- HYBRID AI ENGINE (Code + Text) ---
def ask_ai(question, df):
    if not api_key:
        return "⚠️ API Key missing. Please check your secrets.", None
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # 1. Create Data Context (The "Cheat Sheet" for the AI)
    # This gives the AI a summary of the data so it can answer qualitative questions.
    columns = list(df.columns)
    data_summary = df.describe().to_string()
    
    # 2. Advanced Prompt
    prompt = f"""
    You are a Singapore Housing Data Expert. You have a pandas dataframe 'df'.
    
    --- METADATA ---
    Columns: {columns}
    Data Summary (Stats): 
    {data_summary}
    ----------------
    
    User Question: "{question}"
    
    INSTRUCTIONS:
    1. If the user asks for a specific NUMBER or FACT (e.g., "Average price", "Highest sale"), 
       write a Python code snippet wrapped in ```python ... ``` tags to calculate it.
       The code must return a formatted string (e.g., f"The average is ${{...}}").
       
    2. If the user asks an OPEN-ENDED or ANALYSIS question (e.g., "Explain the price trends", "What data is this?", "Why is Bishan expensive?"), 
       write a normal text response based on the 'Data Summary' above. DO NOT write code.
    
    3. Keep answers concise and professional.
    """
    
    try:
        response = model.generate_content(prompt)
        text_response = response.text.strip()
        
        # 3. Detect if the AI wrote code or text
        # We look for the ```python pattern
        code_match = re.search(r"```python(.*?)```", text_response, re.DOTALL)
        
        if code_match:
            # OPTION A: AI provided Code -> Execute it
            generated_code = code_match.group(1).strip()
            local_vars = {"df": df, "pd": pd}
            
            # Run the code safely
            result = eval(generated_code, {"__builtins__": None}, local_vars)
            return result, generated_code
        else:
            # OPTION B: AI provided Text -> Return it directly
            return text_response, None
            
    except Exception as e:
        return f"I couldn't answer that. (Error: {e})", None

# --- DATA LOADER ---
@st.cache_data(ttl=3600)
def load_all_data():
    if os.path.exists(CACHE_FILE):
        return pd.read_csv(CACHE_FILE)

    base_url = "[https://data.gov.sg/api/action/datastore_search](https://data.gov.sg/api/action/datastore_search)"
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
st.title("🇸🇬 HDB AI Analyst (Open-Ended)")
df = load_all_data()

if not df.empty and 'month' in df.columns:
    df['month'] = pd.to_datetime(df['month'])

if not df.empty:
    # --- AI CHATBOT ---
    st.markdown("### 🤖 Ask Anything")
    st.caption("Try: *'What is the trend?', 'Analyze the prices in Bedok', 'Highest price 4-room flat?'*")
    
    if not api_key:
        st.warning("API Key not found. Please add `GEMINI_API_KEY` to your secrets.")
    else:
        # Chat History container
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display previous chat
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("Ask about HDB trends, facts, or analysis..."):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    answer, code = ask_ai(prompt, df)
                    st.markdown(answer)
                    
                    if code:
                        with st.expander("View Calculation Code"):
                            st.code(code, language="python")
                            
            # Add assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": answer})

    st.divider()

    # --- STANDARD DASHBOARD ---
    st.subheader("📊 Visual Explorer")
    c1, c2 = st.columns(2)
    with c1:
        towns = sorted(df['town'].astype(str).unique())
        sel_towns = st.multiselect("Towns", towns, default=["ANG MO KIO", "BEDOK"])
    with c2:
        types = sorted(df['flat_type'].astype(str).unique())
        default_opts = ['4 ROOM'] if '4 ROOM' in types else [types[0]]
        sel_types = st.multiselect("Types", types, default=default_opts)
    
    if sel_towns and sel_types:
        mask = df['town'].isin(sel_towns) & df['flat_type'].isin(sel_types)
        filt_df = df[mask]
        
        # Map
        stats = filt_df.groupby('town').agg(
            Count=('resale_price', 'count'), Avg=('resale_price', 'mean')
        ).reset_index()
        stats['lat'] = stats['town'].map(lambda x: TOWN_COORDS.get(x, {}).get('lat'))
        stats['lon'] = stats['town'].map(lambda x: TOWN_COORDS.get(x, {}).get('lon'))
        
        if not stats.dropna().empty:
            fig = px.scatter_mapbox(
                stats.dropna(), lat="lat", lon="lon", size="Count", color="Avg",
                color_continuous_scale="Reds", zoom=10, mapbox_style="carto-positron"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        st.dataframe(filt_df.sort_values('month', ascending=False).head(100))