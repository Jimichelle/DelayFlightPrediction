from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DATA = Path(__file__).parents[2] / "app" / "data"
FILES = ["model_scores.csv", "threshold_curves.csv", "correlation.csv"]
FEATURES = {
    "MONTH": "Mois", "DAY": "Jour du mois", "DAY_OF_WEEK": "Jour de la semaine",
    "SCHEDULED_TIME": "Durée prévue", "DISTANCE": "Distance", "dep_hour": "Heure de départ",
    "label": "En retard (> 15 min)",
}


@st.cache_data
def load(name):
    return pd.read_csv(DATA / name)


st.title("🧠 Le modèle")

missing = [f for f in FILES if not (DATA / f).exists()]
if missing:
    st.info(f"Fichiers manquants : {', '.join(missing)}. Relance l'export de `notebooks/flight-delays.ipynb`.")
    st.stop()

scores = load("model_scores.csv").sort_values("auc", ascending=False)
curves = load("threshold_curves.csv")
corr = load("correlation.csv").set_index("Unnamed: 0")  # matrice carrée exportée avec son index

st.markdown(
    "Quatre modèles prédisent si un vol arrivera avec **plus de 15 min de retard**, à partir d'informations "
    "connues avant le départ : compagnie, aéroports, date, heure, durée et distance prévues. "
    "Ils sont entraînés sur 15 % des vols 2015 et évalués sur des vols qu'ils n'ont jamais vus."
)

st.subheader("Scores sur le jeu de test")
st.dataframe(
    scores.rename(columns={"model": "Modèle", "auc": "AUC", "f1": "F1", "accuracy": "Accuracy", "threshold": "Seuil"})
    [["Modèle", "AUC", "F1", "Accuracy", "Seuil"]],
    hide_index=True, width="stretch",
    column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["AUC", "F1", "Accuracy"]}
    | {"Seuil": st.column_config.NumberColumn(format="%.2f")},
)
st.caption(
    "Seuls 18 % des vols sont en retard : un modèle qui répond toujours « à l'heure » aurait 82 % d'accuracy. "
    "On compare donc surtout l'**AUC** (0,5 = hasard, 1 = parfait), qui mesure si le modèle classe les vols "
    "risqués au-dessus des autres, quel que soit le seuil."
)

st.subheader("Effet du seuil de décision")
model = st.selectbox("Modèle", scores["model"])
threshold = float(scores.set_index("model").loc[model, "threshold"])
curve = curves[curves["model"] == model].melt("threshold", ["precision", "recall", "f1"], "Mesure", "Valeur")
curve["Mesure"] = curve["Mesure"].map({"precision": "Précision", "recall": "Rappel", "f1": "F1"})
lines = alt.Chart(curve).mark_line().encode(
    x=alt.X("threshold:Q", title="Seuil de décision"),
    y=alt.Y("Valeur:Q", scale=alt.Scale(domain=[0, 1])),
    color="Mesure:N",
    tooltip=["threshold:Q", "Mesure:N", alt.Tooltip("Valeur:Q", format=".3f")],
)
rule = alt.Chart(pd.DataFrame({"threshold": [threshold]})).mark_rule(strokeDash=[4, 4]).encode(x="threshold:Q")
st.altair_chart(lines + rule, width="stretch")
at = curves[(curves["model"] == model) & (curves["threshold"].round(2) == round(threshold, 2))]
if not at.empty:
    r = at.iloc[0]
    st.caption(
        f"Seuil retenu : **{threshold:.2f}** (pointillés), choisi sur la validation pour maximiser le F1. "
        f"Sur le test, le modèle détecte {r['recall']:.0%} des vols en retard (rappel), "
        f"et {r['precision']:.0%} des vols qu'il annonce en retard le sont vraiment (précision). "
        "Monter le seuil donne moins de fausses alertes mais laisse passer plus de retards."
    )

st.subheader("Matrice de corrélation")
order = [FEATURES[c] for c in corr.columns]
cells = (corr.rename(index=FEATURES, columns=FEATURES)
         .rename_axis("Variable 1").reset_index()
         .melt("Variable 1", var_name="Variable 2", value_name="Corrélation"))
base = alt.Chart(cells).encode(
    x=alt.X("Variable 2:N", sort=order, title=None, axis=alt.Axis(labelAngle=-35)),
    y=alt.Y("Variable 1:N", sort=order, title=None),
)
heatmap = base.mark_rect().encode(
    color=alt.Color("Corrélation:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True)),
    tooltip=["Variable 1:N", "Variable 2:N", alt.Tooltip("Corrélation:Q", format=".3f")],
)
labels = base.mark_text(fontSize=12).encode(
    text=alt.Text("Corrélation:Q", format=".2f"),
    color=alt.condition("abs(datum['Corrélation']) > 0.5", alt.value("white"), alt.value("black")),
)
st.altair_chart((heatmap + labels).properties(height=420), width="stretch")
st.caption(
    "Corrélation de Pearson entre les variables numériques et le retard, sur l'échantillon d'entraînement (15 % des vols). "
    "Les liens linéaires avec le retard sont faibles : c'est pourquoi les modèles non linéaires (XGBoost, réseau de "
    "neurones) font mieux que la régression logistique. Compagnie et aéroports sont des catégories, absentes de la matrice."
)
