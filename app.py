import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import os
import google.generativeai as genai

st.set_page_config(page_title="HDB AI Analyst", layout="wide")

# --- 1. SETUP API KEY ---
# In a real app, use st.secrets. For now, paste your key below or input it in the UI.
# If you want to keep it safe, look at the "Safety Note" below the code.
if "GEMINI_API_KEY" not in st.session_state:
    st.session_state.GEMINI_API_KEY = ""

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

# --- AI ENGINE (Gemini) ---
def ask_ai(question, df, api_key):
    if not api_key:
        return "⚠️ Please enter a Google API Key in the sidebar first."
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    
    # Create the Prompt
    # We teach the AI about your dataframe structure so it can write code for it.
    columns = list(df.columns)
    sample_data = df.head(3).to_string()
    
    prompt = f"""
    You are a Python Data Analyst. 
    You have a pandas dataframe named 'df'.
    
    Here are the columns: {columns}
    Here is a sample of the data:
    {sample_data}
    
    The user asks: "{question}"
    
    Your task:
    1. Write a SINGLE line of Python code using pandas to solve this.
    2. The code must return a result (number, string, or dataframe).
    3. Do NOT use print().
    4. Do NOT wrap the code in markdown blocks (like ```python).
    5. Handle casing: 'town' names are usually uppercase (e.g., 'ANG MO KIO').
    
    Example Question: "Average price in Bedok"
    Example Answer: df[df['town']=='BEDOK']['resale_price'].mean()
    
    Your Answer:
    """
    
    try:
        # Get code from AI
        response = model.generate_content(prompt)
        generated_code = response.text.strip().replace("```python", "").replace("```", "")
        
        # Execute the code safely
        # We use a local scope where 'df' and 'pd' are available
        local_vars = {"df": df, "pd": pd}
        result = eval(generated_code, {"__builtins__": None}, local_vars)
        
        return result, generated_code
    except Exception as e:
        return f"Sorry, I couldn't calculate that. (Error: {e})", None


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
st.title("🇸🇬 HDB AI Analyst")

# Sidebar for API Key
with st.sidebar:
    st.header("🔑 AI Setup")
    api_input = st.text_input("Enter Google Gemini API Key", type="password")
    if api_input:
        st.session_state.GEMINI_API_KEY = api_input
    st.caption("[Get a free key here](https://aistudio.google.com/app/apikey)")
    st.divider()

# Load Data
df = load_all_data()

if not df.empty and 'month' in df.columns:
    df['month'] = pd.to_datetime(df['month'])

if not df.empty:
    # --- AI CHATBOT SECTION ---
    st.markdown("### 🤖 Ask the Data")
    st.info("This AI writes Python code to answer your questions accurately.")
    
    question = st.text_input("Ask a question about HDB prices:", placeholder="e.g., 'What is the most expensive 5 room flat in Ang Mo Kio?'")
    
    if question:
        if not st.session_state.GEMINI_API_KEY:
            st.error("Please enter your API Key in the sidebar to use the AI.")
        else:
            with st.spinner("Analyzing data..."):
                answer, code = ask_ai(question, df, st.session_state.GEMINI_API_KEY)
                
                # Display Result
                st.success(f"**Answer:** {answer}")
                
                # Show the 'thinking' process (optional transparency)
                with st.expander("See how I calculated this (Python Code)"):
                    st.code(code, language="python")

    st.divider()

    # --- STANDARD DASHBOARD ---
    st.subheader("📊 Visual Explorer")
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        towns = sorted(df['town'].astype(str).unique())
        sel_towns = st.multiselect("Towns", towns, default=["ANG MO KIO", "BEDOK"])
    with col2:
        types = sorted(df['flat_type'].astype(str).unique())
        sel_types = st.multiselect("Flat Types", types, default=['4 ROOM'])
    
    if sel_towns and sel_types:
        mask = df['town'].isin(sel_towns) & df['flat_type'].isin(sel_types)
        filt_df = df[mask]
        
        # Map
        stats = filt_df.groupby('town').agg(
            Count=('resale_price', 'count'),
            Avg=('resale_price', 'mean')
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