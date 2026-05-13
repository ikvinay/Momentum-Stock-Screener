"""
Commodities page — placeholder for future implementation.
"""

import streamlit as st
from ui.components import inject_css
from ui.sidebar import render_data_management

inject_css()

with st.sidebar:
    render_data_management()

st.markdown("# 🏗️ Commodities")
st.divider()

st.info(
    "**Coming soon.**\n\n"
    "This section will track major commodity indices and futures traded on Indian exchanges:\n\n"
    "- **MCX Gold / Silver** — precious metals momentum\n"
    "- **MCX Crude Oil / Natural Gas** — energy\n"
    "- **NCDEX Agricultural** — Agri commodities\n"
    "- **MCX iCOMDEX** — broad commodity index\n\n"
    "Ranking will follow the same scoring system as Stocks and Indices, with RS "
    "measured against the MCX iCOMDEX composite."
)
