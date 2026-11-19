"""Dashboard Streamlit – Retards de vols US 2015 (étape 4.7).

Lit uniquement les tables exportées par notebooks/flight-delays.ipynb dans app/data/ :
  - stats_airline.csv     AIRLINE, flights, avg_arr_delay, delay_rate_pct
  - stats_hour.csv        dep_hour, flights, avg_arr_delay
  - model_scores.csv      model, accuracy, f1, auc
  - predictions.csv       (optionnel) label, prediction
Lancer depuis la racine du repo :  streamlit run app/app.py
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"
AIRLINES_CSV = Path(__file__).parents[1] / "data" / "airlines.csv"  # noms des compagnies (optionnel)

st.set_page_config(page_title="Retards de vols US 2015", page_icon="✈️", layout="wide")


@st.cache_data
def load(name: str) -> pd.DataFrame | None:
    path = DATA_DIR / name
    return pd.read_csv(path) if path.exists() else None


def missing(name: str) -> None:
    st.info(f"`app/data/{name}` introuvable : exécuter le notebook jusqu'à la cellule d'export.")


@st.cache_data
def airline_names() -> dict[str, str]:
    if AIRLINES_CSV.exists():
        df = pd.read_csv(AIRLINES_CSV)
        return dict(zip(df["IATA_CODE"], df["AIRLINE"]))
    return {}


airlines = load("stats_airline.csv")
hours = load("stats_hour.csv")
scores = load("model_scores.csv")
preds = load("predictions.csv")

# ── En-tête ──────────────────────────────────────────────────────────────────
st.title("✈️ Retards de vols aux États-Unis – 2015")
st.caption(
    "Dataset USDOT 2015 Flight Delays (Kaggle) · stocké sur HDFS · analysé et modélisé avec PySpark. "
    "Un vol est **en retard** si `ARRIVAL_DELAY > 15 min`."
)

# ── Chiffres clés ────────────────────────────────────────────────────────────
st.header("Chiffres clés")
c1, c2, c3, c4 = st.columns(4)
if airlines is not None:
    total = int(airlines["flights"].sum())
    delay_rate = (airlines["delay_rate_pct"] * airlines["flights"]).sum() / total
    avg_delay = (airlines["avg_arr_delay"] * airlines["flights"]).sum() / total
    c1.metric("Vols analysés", f"{total:,}".replace(",", " "))
    c2.metric("Taux de vols en retard", f"{delay_rate:.1f} %")
    c3.metric("Retard moyen à l'arrivée", f"{avg_delay:.1f} min")
if scores is not None:
    best = scores.sort_values("auc", ascending=False).iloc[0]
    c4.metric(f"Meilleur modèle · AUC {best['auc']:.3f}", best["model"])

# ── Compagnies ───────────────────────────────────────────────────────────────
st.header("Par compagnie")
if airlines is None:
    missing("stats_airline.csv")
else:
    df = airlines.copy()
    df["compagnie"] = df["AIRLINE"].map(airline_names()).fillna(df["AIRLINE"])
    metric = st.radio(
        "Indicateur", ["delay_rate_pct", "avg_arr_delay"], horizontal=True,
        format_func={"delay_rate_pct": "Taux de retard (%)", "avg_arr_delay": "Retard moyen (min)"}.get,
    )
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{metric}:Q", title=None),
            y=alt.Y("compagnie:N", sort="-x", title=None),
            tooltip=["compagnie", "flights", "avg_arr_delay", "delay_rate_pct"],
        )
    )
    st.altair_chart(chart, width="stretch")
    with st.expander("Données"):
        st.dataframe(df, hide_index=True, width="stretch")

# ── Heure de départ ──────────────────────────────────────────────────────────
st.header("Par heure de départ prévue")
if hours is None:
    missing("stats_hour.csv")
else:
    left, right = st.columns(2)
    left.subheader("Retard moyen (min)")
    left.line_chart(hours, x="dep_hour", y="avg_arr_delay")
    right.subheader("Nombre de vols")
    right.bar_chart(hours, x="dep_hour", y="flights")
    st.caption("Les retards s'accumulent au fil de la journée : effet cascade des avions en retard.")  # à valider sur les données

# ── Modèles ──────────────────────────────────────────────────────────────────
st.header("Modèles de prédiction")
if scores is None:
    missing("model_scores.csv")
else:
    st.dataframe(scores.style.format({"accuracy": "{:.3f}", "f1": "{:.3f}", "auc": "{:.3f}"}),
                 hide_index=True, width="stretch")
    if airlines is not None:
        baseline = 1 - delay_rate / 100
        st.caption(
            f"Référence : prédire toujours « à l'heure » donne une accuracy de **{baseline:.3f}**. "
            "L'AUC est donc la métrique à regarder (0,5 = hasard)."
        )
    long = scores.melt(id_vars="model", var_name="métrique", value_name="score")
    st.altair_chart(
        alt.Chart(long).mark_bar().encode(
            x=alt.X("métrique:N", title=None), y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1])),
            color="model:N", xOffset="model:N", tooltip=["model", "métrique", alt.Tooltip("score", format=".3f")],
        ),
        width="stretch",
    )

# ── Prédictions (optionnel) ──────────────────────────────────────────────────
if preds is not None:
    st.header("Matrice de confusion (test set)")
    cm = pd.crosstab(preds["label"], preds["prediction"], rownames=["Réel"], colnames=["Prédit"])
    st.dataframe(cm, width="stretch")
