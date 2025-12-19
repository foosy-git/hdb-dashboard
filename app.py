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
    
    # Simple progress bar
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

# --- HELPER: GENERATE SMART CONTEXT ---
def get_dataset_context(df):
    """Creates a 'Cheat Sheet' for the AI with aggregated stats."""
    
    # 1. Yearly Trend
    df['year'] = df['month'].dt.year
    yearly_trend = df.groupby('year')['resale_price'].mean().to_dict()
    
    # 2. Town Averages (Sorted)
    town_avg = df.groupby('town')['resale_price'].mean().sort_values(ascending=False)
    most_expensive = town_avg.head(5).to_dict()
    least_expensive = town_avg.tail(5).to_dict()
    
    # 3. Flat Type Averages
    type_avg = df.groupby('flat_type')['resale_price'].mean().to_dict()
    
    summary = f"""
    --- DATA CHEAT SHEET ---
    [Yearly Average Prices]: {yearly_trend}
    [Most Expensive Towns]: {most_expensive}
    [Cheapest Towns]: {least_expensive}
    [Prices by Flat Type]: {type_avg}
    ------------------------
    """
    return summary

# --- AI ENGINE ---
def ask_ai(question, df):
    if not api_key:
        return "⚠️ API Key missing.", None
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Give the AI the Smart Context
    context_str = get_dataset_context(df)
    columns = list(df.columns)
    
    prompt = f"""
    You are a Singapore Property Expert.
    
    {context_str}
    
    Columns: {columns}
    User Question: "{question}"
    
    INSTRUCTIONS:
    1. For ANALYSIS (e.g., "Why is Bishan expensive?", "Trend analysis"): Use the [Most Expensive Towns] and [Yearly Average Prices] data provided above to explain. Do NOT write code. Write a helpful paragraph.
    
    2. For RECOMMENDATIONS (e.g., "I have 400k, where to buy?"): Look at the [Cheapest Towns] list above. Suggest specific towns that fit the budget. Do NOT write code.
    
    3. For SPECIFIC CALCULATIONS (e.g., "Exact average price in Bedok"): Write Python code wrapped in ```python ... ```.
    
    """
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Check for code
        code_match = re.search(r"```python(.*?)```", text, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            local_vars = {"df": df, "pd": pd}
            result = eval(code, {"__builtins__": None}, local_vars)
            return result, code
        else:
            return text, None
            
    except Exception as e:
        return f"Error: {e}", None

# --- MAIN APP UI ---
st.title("🇸🇬 HDB AI Analyst")

df = load_all_data()
if not df.empty and 'month' in df.columns:
    df['month'] = pd.to_datetime(df['month'])

if not df.empty:
    
    # --- GLOBAL FILTERS (SIDEBAR) ---
    st.sidebar.header("Filter Settings")
    
    # Town Filter
    all_towns = sorted(df['town'].astype(str).unique())
    sel_towns = st.sidebar.multiselect("Select Towns", all_towns, default=["ANG MO KIO", "BEDOK"])
    
    # Flat Type Filter
    all_types = sorted(df['flat_type'].astype(str).unique())
    default_types = ['4 ROOM'] if '4 ROOM' in all_types else [all_types[0]]
    sel_types = st.sidebar.multiselect("Flat Types", all_types, default=default_types)
    
    # Apply Filters
    # (If nothing selected, show everything to avoid empty charts)
    if not sel_towns: sel_towns = all_towns
    if not sel_types: sel_types = all_types
    
    mask = df['town'].isin(sel_towns) & df['flat_type'].isin(sel_types)
    filt_df = df[mask]

    # --- 1. VISUAL DASHBOARD ---
    # We place this FIRST so it's always visible
    st.subheader("📊 Visual Explorer")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected Volume", f"{len(filt_df):,}")
    c2.metric("Avg Price", f"${filt_df['resale_price'].mean():,.0f}")
    c3.metric("Max Price", f"${filt_df['resale_price'].max():,.0f}")
    
    # TABS for Charts
    tab1, tab2 = st.tabs(["📈 Trend Chart", "🗺️ Map View"])
    
    with tab1:
        if not filt_df.empty:
            # Group by Month & Town for multi-line chart
            trend_data = filt_df.groupby(['month', 'town'])['resale_price'].mean().reset_index().sort_values('month')
            fig_line = px.line(trend_data, x='month', y='resale_price', color='town', title="Price Trend Over Time")
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No data for trend chart.")

    with tab2:
        if not filt_df.empty:
            town_stats = filt_df.groupby('town').agg(
                Count=('resale_price', 'count'), Avg=('resale_price', 'mean')
            ).reset_index()
            # Add Coords
            town_stats['lat'] = town_stats['town'].map(lambda x: TOWN_COORDS.get(x, {}).get('lat'))
            town_stats['lon'] = town_stats['town'].map(lambda x: TOWN_COORDS.get(x, {}).get('lon'))
            
            if not town_stats.dropna().empty:
                fig_map = px.scatter_mapbox(
                    town_stats.dropna(), lat="lat", lon="lon", size="Count", color="Avg",
                    color_continuous_scale="Reds", zoom=10, mapbox_style="carto-positron"
                )
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.warning("Map coordinates missing.")

    st.divider()

    # --- 2. AI CHATBOT ---
    st.subheader("🤖 AI Consultant")
    st.caption("Ask: 'Where can I find a 400k flat?', 'Why is Bishan expensive?', 'Analyze the trend'")

    # Chat UI
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Pass FULL dataframe to AI so it knows global context
                answer, code = ask_ai(prompt, df) 
                st.write(answer)
                if code:
                    with st.expander("Show Calculation"):
                        st.code(code)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.error("Data could not be loaded.")