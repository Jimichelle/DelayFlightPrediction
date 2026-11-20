"""Chemins et aides partagés par les pages de l'app."""
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parents[1]
DATA = ROOT / "app" / "data"  # CSV exportés par le notebook
MODELS = ROOT / "app" / "models"  # modèles et preprocessing.json exportés par le notebook
RAW = ROOT / "data"  # CSV Kaggle (noms des compagnies et des aéroports)
RERUN = "Relance l'export de `notebooks/flight-delays.ipynb`."


def require(*paths):
    """Affiche un message et arrête la page si un fichier exporté par le notebook manque."""
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        st.info(f"Fichiers manquants : {', '.join(missing)}. {RERUN}")
        st.stop()


@st.cache_data
def airline_names():
    path = RAW / "airlines.csv"
    return dict(pd.read_csv(path).values) if path.exists() else {}
