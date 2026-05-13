"""
Momentum Screener — entry point.
Run with:  streamlit run app.py
"""

import logging
import sys
import streamlit as st

st.set_page_config(
    page_title="Momentum Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
)

from src.scheduler import start_scheduler
start_scheduler()

pg = st.navigation([
    st.Page("pages/home.py",          title="Home",        icon="🏠", default=True),
    st.Page("pages/1_Stocks.py",      title="Stocks",      icon="📊"),
    st.Page("pages/2_Indices.py",     title="NSE Indices", icon="🗂️"),
    st.Page("pages/3_Commodities.py", title="Commodities", icon="🏗️"),
])
pg.run()
