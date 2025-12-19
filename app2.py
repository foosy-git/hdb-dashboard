import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import os

st.set_page_config(page_title="HDB Analytics Pro", layout="wide")

# --- CONFIGURATION ---
RESOURCE_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
CACHE_FILE = "hdb_full_data_cache.csv"

# --- STATIC DATA: HDB TOWN COORDINATES ---
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
    
    # Progress UI
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        while True:
            params = {
                "resource_id": RESOURCE_ID,
                "limit": limit,
                "offset": offset,
                "sort": "month desc"
            }
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if not data.get('success'): break
            records = data['result']['records']
            if not records: break
                
            all_records.extend(records)
            status_text.text(f"Downloaded {len(all_records):,} records...")
            
            if len(records) < limit: break
            offset += limit
            if len(all_records) > 300000: break 

        progress_bar.empty()
        status_text.empty()
        
        df = pd.DataFrame(all_records)
        # Type conversion
        if 'month' in df.columns: df['month'] = pd.to_datetime(df['month'])
        cols = ['resale_price', 'floor_area_sqm', 'lease_commence_date']
        for c in cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

        df.to_csv(CACHE_FILE, index=False)
        return df

    except Exception as e:
        st.error(f"Download Error: {e}")
        return pd.DataFrame()

# --- MAIN APP ---
st.title("🇸🇬 Unified HDB Dashboard")

with st.spinner("Loading Database..."):
    df = load_all_data()
    if not df.empty and 'month' in df.columns:
        df['month'] = pd.to_datetime(df['month'])

if not df.empty:
    # --- 1. GLOBAL SIDEBAR FILTERS ---
    st.sidebar.header("Global Filters")
    st.sidebar.info("Filters update Map, Chart, and Table.")
    
    # Date Filter
    min_date = df['month'].min().date()
    max_date = df['month'].max().date()
    st.sidebar.subheader("1. Date Range")
    start_date, end_date = st.sidebar.date_input("Select Period", (min_date, max_date), min_value=min_date, max_value=max_date)

    # Town Filter
    st.sidebar.subheader("2. Select Towns")
    all_towns = sorted(df['town'].astype(str).unique())
    selected_towns = st.sidebar.multiselect("Select Towns", all_towns, default=["ANG MO KIO", "BEDOK", "CLEMENTI"])

    # Flat Type Filter (UPDATED to Multiselect)
    st.sidebar.subheader("3. Flat Type")
    flat_types = sorted(df['flat_type'].astype(str).unique())
    # Default to '4 ROOM' if available
    default_opts = ['4 ROOM'] if '4 ROOM' in flat_types else [flat_types[0]]
    selected_flats = st.sidebar.multiselect("Choose Types", flat_types, default=default_opts)

    # --- 2. APPLY FILTERS ---
    if not selected_towns or not selected_flats:
        st.warning("Please select at least one Town and one Flat Type.")
    else:
        mask_date = (df['month'].dt.date >= start_date) & (df['month'].dt.date <= end_date)
        mask_town = (df['town'].isin(selected_towns))
        mask_type = (df['flat_type'].isin(selected_flats)) # Updated to .isin()
        
        filtered_df = df[mask_date & mask_town & mask_type].copy()

        if filtered_df.empty:
            st.warning("No data found for the selected combination.")
        else:
            # --- 3. METRICS ---
            st.markdown("### 📊 Market Snapshot")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Selected Transactions", f"{len(filtered_df):,}")
            col2.metric("Average Price", f"${filtered_df['resale_price'].mean():,.0f}")
            col3.metric("Lowest Price", f"${filtered_df['resale_price'].min():,.0f}")
            col4.metric("Highest Price", f"${filtered_df['resale_price'].max():,.0f}")

            # --- 4. MAP VISUALIZATION (RED SCALE) ---
            st.divider()
            st.subheader(f"🗺️ Geographic Distribution (Darker Red = More Expensive)")
            
            # Aggregate stats
            town_stats = filtered_df.groupby('town').agg(
                Count=('resale_price', 'count'),
                Avg_Price=('resale_price', 'mean'),
                Min_Price=('resale_price', 'min'),
                Max_Price=('resale_price', 'max')
            ).reset_index()

            # Add GPS
            town_stats['lat'] = town_stats['town'].map(lambda x: TOWN_COORDS.get(x, {}).get('lat'))
            town_stats['lon'] = town_stats['town'].map(lambda x: TOWN_COORDS.get(x, {}).get('lon'))
            town_stats = town_stats.dropna(subset=['lat', 'lon'])

            if not town_stats.empty:
                fig_map = px.scatter_mapbox(
                    town_stats,
                    lat="lat",
                    lon="lon",
                    size="Count", 
                    color="Avg_Price",
                    # UPDATED COLOR SCALE: "Reds"
                    color_continuous_scale="Reds", 
                    size_max=40, zoom=10,
                    mapbox_style="carto-positron",
                    hover_name="town",
                    hover_data={"lat":False, "lon":False, "Avg_Price":":,.0f", "Count":":,"},
                    title="Bubble Size = Volume | Color Intensity = Price"
                )
                st.plotly_chart(fig_map, use_container_width=True)
            
            # --- 5. TREND CHART ---
            st.divider()
            st.subheader("📈 Price Trends")
            
            # Group by Month & Town
            trend_df = filtered_df.groupby(['month', 'town'])['resale_price'].mean().reset_index().sort_values('month')
            
            fig_line = px.line(
                trend_df, x='month', y='resale_price', color='town', markers=True,
                title=f"Monthly Average Price (Combined: {', '.join(selected_flats)})",
                labels={'resale_price': 'Price (SGD)', 'month': 'Date'}
            )
            st.plotly_chart(fig_line, use_container_width=True)

            # --- 6. TABLE ---
            st.divider()
            st.subheader("📋 Transaction Search")
            
            search_query = st.text_input("🔍 Filter (Street Name or Block):", "")
            
            if search_query:
                search_mask = (
                    filtered_df['street_name'].str.contains(search_query, case=False, na=False) |
                    filtered_df['block'].str.contains(search_query, case=False, na=False)
                )
                display_df = filtered_df[search_mask]
            else:
                display_df = filtered_df

            st.dataframe(
                display_df.sort_values('month', ascending=False),
                column_config={
                    "month": st.column_config.DateColumn("Month"),
                    "resale_price": st.column_config.NumberColumn("Price", format="$%d"),
                    "remaining_lease": st.column_config.TextColumn("Lease Left")
                },
                height=400
            )

else:
    st.error("Unable to load data.")