"""Home page — 3-tile navigation hub."""

import streamlit as st
from ui.components import inject_css
from ui.sidebar import render_data_management
from src.pipeline import load_results, load_index_results

inject_css()

with st.sidebar:
    render_data_management()

st.markdown("# 📈 Momentum Screener")
st.caption("NSE Nifty 1000 · EMA Stack · VCP · Sector RS · Daily Auto-Refresh at 16:00 IST")
st.divider()

results       = load_results()
index_results = load_index_results()
n_stocks      = len(results)       if results       is not None else 0
n_indices     = len(index_results) if index_results is not None else 0

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    with st.container(border=True):
        st.markdown("## 📊")
        st.markdown("### Stocks")
        st.write("NSE Nifty 1000 · EMA Stack · VCP · Sector RS")
        st.metric("Passing today", n_stocks)
        if st.button("Open Stocks →", key="tile_stocks", use_container_width=True):
            st.switch_page("pages/1_Stocks.py")

with col2:
    with st.container(border=True):
        st.markdown("## 🗂️")
        st.markdown("### NSE Indices")
        st.write("52 indices · EMA10 > EMA20 > EMA50 · RS vs Nifty 500")
        st.metric("Passing today", n_indices)
        if st.button("Open Indices →", key="tile_indices", use_container_width=True):
            st.switch_page("pages/2_Indices.py")

with col3:
    with st.container(border=True):
        st.markdown("## 🏗️")
        st.markdown("### Commodities")
        st.write("MCX Gold, Silver, Crude, Agri · Coming soon")
        st.metric("Passing today", "—")
        if st.button("Preview →", key="tile_commodities", use_container_width=True):
            st.switch_page("pages/3_Commodities.py")
