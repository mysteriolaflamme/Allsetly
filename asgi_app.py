"""One deployment, two front doors: the Streamlit app at / and the business API at /api.

Run with ``streamlit run asgi_app.py`` (API docs at /api/docs). ``streamlit run app.py``
still serves the app alone.
"""

import streamlit as st
from starlette.routing import Mount

from temoin.api import api

app = st.App("app.py", routes=[Mount("/api", app=api)])
