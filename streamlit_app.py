"""Superstore Profit Intelligence dashboard (Streamlit entry point).

Run locally from the project root:
    streamlit run streamlit_app.py

On Streamlit Community Cloud this file is the app's main file. On first start the
app builds the DuckDB database from data/raw/superstore.csv and trains the
loss-risk model; both are cached afterwards.
"""

import streamlit as st

from app.views import customers, overview, risk_checker

st.set_page_config(page_title="Superstore Profit Intelligence", page_icon="📊", layout="wide")

pages = st.navigation([
    st.Page(overview.render, title="Overview", url_path="overview", default=True),
    st.Page(customers.render, title="Customers", url_path="customers"),
    st.Page(risk_checker.render, title="Loss-risk checker", url_path="risk-checker"),
])
st.sidebar.caption("Data: Sample Superstore (2014-2017). "
                   "[Source code](https://github.com/orelrubin2/superstore-profit-intelligence)")
pages.run()
