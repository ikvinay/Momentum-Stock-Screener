"""
Market Insights — entry point.
Run with:  streamlit run app.py
"""

import base64
import logging
import sys
import streamlit as st

st.set_page_config(
    page_title="Market Insights",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Render app title in the native Streamlit sidebar header (stSidebarHeader)
# via st.logo() so it uses Streamlit's own logo mechanism, not CSS injection.
_LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="240" height="50">'
    '<text x="0" y="38"'
    ' font-family="-apple-system,BlinkMacSystemFont,\'Segoe UI\',sans-serif"'
    ' font-size="27" font-weight="700" fill="#f1f5f9" letter-spacing="-0.4">'
    'Market Insights'
    '</text>'
    '</svg>'
)
_LOGO_B64 = base64.b64encode(_LOGO_SVG.encode()).decode()
st.logo(f"data:image/svg+xml;base64,{_LOGO_B64}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
)

from ui.components import inject_css
inject_css()

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
