"""Application Streamlit – Retards de vols US 2015.

Point d'entrée multipage : définit la navigation, les pages vivent dans app/pages/.
Lancer depuis la racine du repo :  streamlit run app/app.py
"""
import streamlit as st

st.set_page_config(page_title="Retards de vols US 2015", layout="wide")

nav = st.navigation([
    st.Page("pages/predict.py", title="Prédire un retard", default=True),
    st.Page("pages/explore.py", title="Explorer les retards"),
    st.Page("pages/model.py", title="Le modèle"),
])
nav.run()
