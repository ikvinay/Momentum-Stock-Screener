"""
Market Insights — entry point.
Run with:  streamlit run app.py
"""

import logging
import sys
import streamlit as st

st.set_page_config(
    page_title="Market Insights",
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
    st.Page("pages/home.py",          title="Home",        icon=":material/home:",              default=True),
    st.Page("pages/1_Stocks.py",      title="Stocks",      icon=":material/candlestick_chart:"),
    st.Page("pages/2_Indices.py",     title="Themes",      icon=":material/stacked_line_chart:"),
    st.Page("pages/3_Commodities.py", title="Commodities", icon=":material/diamond:"),
    st.Page("pages/Admin.py",         title="Admin",       icon=":material/admin_panel_settings:"),
])
pg.run()
