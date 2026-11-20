from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parents[2]
DETAIL_CSV = ROOT / "app" / "data" / "stats_detail.csv"
AIRLINES_CSV = ROOT / "data" / "airlines.csv"
MONTHS = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]


@st.cache_data
def load_detail():
    return pd.read_csv(DETAIL_CSV)


@st.cache_data
def airline_names():
    return dict(pd.read_csv(AIRLINES_CSV).values) if AIRLINES_CSV.exists() else {}


def aggregate(df, by):
    """Taux et moyennes recalculés à partir des sommes : exacts quel que soit le filtre."""
    g = df.groupby(by, as_index=False)[["flights", "delayed", "sum_arr_delay"]].sum()
    g["Taux de retard (%)"] = (100 * g["delayed"] / g["flights"]).round(1)
    g["Retard moyen (min)"] = (g["sum_arr_delay"] / g["flights"]).round(1)
    return g


st.title("📊 Explorer les retards")
st.caption("Vols 2015 nettoyés (hors annulés et déroutés). Un vol est « en retard » s'il arrive plus de 15 min après l'horaire prévu.")

if not DETAIL_CSV.exists():
    st.info("`stats_detail.csv` manquant. Relance l'export de `notebooks/temp_notebook.ipynb`.")
    st.stop()

detail = load_detail()
names = airline_names()

f1, f2, f3 = st.columns([2, 1, 1])
airlines = f1.multiselect("Compagnies (toutes si vide)", sorted(detail["AIRLINE"].unique()),
                          format_func=lambda c: f"{c} – {names.get(c, c)}")
months = f2.select_slider("Mois", options=list(range(1, 13)), value=(1, 12), format_func=lambda m: MONTHS[m - 1])
hours = f3.slider("Heure de départ", 0, 23, (0, 23), format="%dh")
metric = st.radio("Indicateur", ["Taux de retard (%)", "Retard moyen (min)"], horizontal=True)

df = detail[detail["MONTH"].between(*months) & detail["dep_hour"].between(*hours)]
if airlines:
    df = df[df["AIRLINE"].isin(airlines)]
if df.empty:
    st.warning("Aucun vol pour ces filtres.")
    st.stop()

total = aggregate(df.assign(all=1), "all").iloc[0]
k1, k2, k3 = st.columns(3)
k1.metric("Vols", f"{int(total['flights']):,}".replace(",", " "))
k2.metric("Taux de retard", f"{total['Taux de retard (%)']:.1f} %")
k3.metric("Retard moyen à l'arrivée", f"{total['Retard moyen (min)']:.1f} min")

st.subheader("Par compagnie")
by_airline = aggregate(df, "AIRLINE")
by_airline["Compagnie"] = by_airline["AIRLINE"].map(lambda c: names.get(c, c))
st.bar_chart(by_airline.sort_values(metric, ascending=False), x="Compagnie", y=metric, horizontal=True, sort=f"-{metric}")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Par heure de départ")
    by_hour = aggregate(df, "dep_hour").rename(columns={"dep_hour": "Heure"})
    st.line_chart(by_hour, x="Heure", y=metric)
with c2:
    st.subheader("Par mois")
    by_month = aggregate(df, "MONTH")
    by_month["Mois"] = by_month["MONTH"].map(lambda m: f"{m:02d} {MONTHS[m - 1]}")  # préfixe numérique : garde l'ordre
    st.bar_chart(by_month, x="Mois", y=metric)

with st.expander("Données par compagnie"):
    st.dataframe(
        by_airline[["AIRLINE", "Compagnie", "flights", "Taux de retard (%)", "Retard moyen (min)"]]
        .rename(columns={"AIRLINE": "Code", "flights": "Vols"})
        .sort_values("Taux de retard (%)", ascending=False),
        hide_index=True, width="stretch",
    )
