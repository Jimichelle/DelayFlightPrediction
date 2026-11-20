import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parents[2]
MODEL_DIR = ROOT / "app" / "models"
ROUTES_CSV = ROOT / "app" / "data" / "routes.csv"
AIRLINES_CSV = ROOT / "data" / "airlines.csv"
AIRPORTS_CSV = ROOT / "data" / "airports.csv"


@st.cache_resource
def load_models(files):
    # imports lents : seulement si les modèles existent
    import tensorflow as tf
    import xgboost as xgb

    models = {}
    for f in files:
        if f.endswith("_spark"):  # dossier Spark MLlib : demande une session Spark locale (Java)
            from pyspark.ml.classification import RandomForestClassificationModel
            from pyspark.sql import SparkSession

            (SparkSession.builder.master("local[1]").appName("flight-delays-app")
             .config("spark.ui.enabled", "false").getOrCreate())
            models[f] = RandomForestClassificationModel.load(str(MODEL_DIR / f))
        elif f.endswith(".json"):  # XGBoost (Booster natif)
            models[f] = xgb.Booster(model_file=str(MODEL_DIR / f))
        else:
            models[f] = tf.keras.models.load_model(MODEL_DIR / f)
    return models


def n_inputs(model):
    if hasattr(model, "numFeatures"):  # Spark
        return model.numFeatures
    if hasattr(model, "num_features"):  # XGBoost
        return model.num_features()
    return model.input_shape[1]  # Keras


def delay_proba(model, x):
    if hasattr(model, "predictProbability"):  # Spark : un seul vecteur, sans DataFrame
        from pyspark.ml.linalg import Vectors

        return float(model.predictProbability(Vectors.dense(x[0].tolist()))[1])
    if hasattr(model, "num_features"):  # XGBoost
        import xgboost as xgb

        return float(model.predict(xgb.DMatrix(x))[0])
    return float(model.predict(x, verbose=0)[0, 0])  # Keras


@st.cache_data
def load_preprocessing():
    return json.loads((MODEL_DIR / "preprocessing.json").read_text())


@st.cache_data
def load_routes():
    """Une ligne par (départ, arrivée, compagnie) avec durée et distance médianes."""
    return pd.read_csv(ROUTES_CSV)


@st.cache_data
def airline_names():
    return dict(pd.read_csv(AIRLINES_CSV).values) if AIRLINES_CSV.exists() else {}


@st.cache_data
def airport_names():
    if not AIRPORTS_CSV.exists():
        return {}
    df = pd.read_csv(AIRPORTS_CSV).dropna(subset=["CITY", "STATE"])
    return dict(zip(df["IATA_CODE"], df["CITY"] + ", " + df["STATE"]))


def category_index(labels, value):
    return labels.index(value) if value in labels else len(labels)  # len(labels) = valeur inconnue


def encode(prep, values, inputs="onehot"):
    """Reproduit le notebook.
    onehot : numériques standardisés puis one-hot (to_features, Keras et XGBoost).
    raw    : numériques bruts puis index de catégorie (rf_assembler, Random Forest Spark)."""
    nums = np.array([values[c] for c in prep["num_cols"]], dtype=np.float32)
    cats = [(labels, category_index(labels, values[col])) for col, labels in zip(prep["cat_cols"], prep["cat_labels"])]
    if inputs == "raw":
        return np.concatenate([nums, [i for _, i in cats]]).astype(np.float32)[None, :]
    parts = [(nums - np.array(prep["mean"], dtype=np.float32)) / np.array(prep["std"], dtype=np.float32)]
    for labels, i in cats:
        ohe = np.zeros(len(labels) + 1, dtype=np.float32)
        ohe[i] = 1.0
        parts.append(ohe)
    return np.concatenate(parts)[None, :]


st.title("🔮 Prédire un retard")

RERUN = "Relance l'export de `notebooks/temp_notebook.ipynb`."
missing = [p.name for p in (MODEL_DIR / "preprocessing.json", ROUTES_CSV) if not p.exists()]
if missing:
    st.info(f"Fichiers manquants : {', '.join(missing)}. {RERUN}")
    st.stop()

routes = load_routes()
prep = load_preprocessing()
if "AIRLINE" not in routes.columns or "models" not in prep:
    st.info(f"`routes.csv` ou `preprocessing.json` date d'un ancien export. {RERUN}")
    st.stop()
missing = [m["file"] for m in prep["models"] if not (MODEL_DIR / m["file"]).exists()]
if missing:
    st.info(f"Modèles manquants : {', '.join(missing)}. {RERUN}")
    st.stop()

models = load_models(tuple(m["file"] for m in prep["models"]))
airlines, airports = airline_names(), airport_names()
airport_label = lambda code: f"{code} – {airports[code]}" if code in airports else code

# chaque liste ne propose que ce qui existe vraiment dans les vols 2015
origin = st.selectbox("Départ", sorted(routes["ORIGIN_AIRPORT"].unique()), format_func=airport_label)
from_origin = routes[routes["ORIGIN_AIRPORT"] == origin]
dest = st.selectbox("Arrivée", sorted(from_origin["DESTINATION_AIRPORT"].unique()), format_func=airport_label)
on_route = from_origin[from_origin["DESTINATION_AIRPORT"] == dest].sort_values("flights", ascending=False)
airline = st.selectbox("Compagnie", on_route["AIRLINE"], format_func=lambda c: f"{c} – {airlines.get(c, c)}")

c1, c2 = st.columns(2)
date = c1.date_input("Date du vol", dt.date.today(), min_value=dt.date.today(),
                     help="Le modèle utilise le mois, le jour et le jour de la semaine, pas l'année (appris sur 2015).")
hour = c2.slider("Heure de départ", 0, 23, 17, format="%dh")

route = on_route.set_index("AIRLINE").loc[airline]
sched_time, distance = int(route["sched_time"]), int(route["distance"])
st.caption(f"{origin} → {dest} · {distance:,} miles · {sched_time // 60}h{sched_time % 60:02d} prévues · "
           f"{int(route['flights']):,} vols de cette compagnie en 2015")

if st.button("Prédire", type="primary"):
    values = {
        "MONTH": date.month, "DAY": date.day, "DAY_OF_WEEK": date.isoweekday(),  # 1 = lundi, comme le dataset
        "SCHEDULED_TIME": sched_time, "DISTANCE": distance, "dep_hour": hour,
        "AIRLINE": airline, "ORIGIN_AIRPORT": origin, "DESTINATION_AIRPORT": dest,
    }
    xs = {m["file"]: encode(prep, values, m.get("inputs", "onehot")) for m in prep["models"]}
    wrong = [m["name"] for m in prep["models"] if n_inputs(models[m["file"]]) != xs[m["file"]].shape[1]]
    if wrong:
        st.error(f"preprocessing.json ne correspond pas à : {', '.join(wrong)}. {RERUN}")
        st.stop()

    st.subheader("Probabilité de retard > 15 min")
    for col, m in zip(st.columns(len(prep["models"])), prep["models"]):  # meilleure AUC en premier
        proba = delay_proba(models[m["file"]], xs[m["file"]])
        verdict = "En retard" if proba >= m["threshold"] else "À l'heure"
        col.metric(m["name"], f"{proba:.0%}")
        col.progress(proba)
        col.caption(f"{verdict} · seuil {m['threshold']:.2f} · AUC test {m['auc']:.3f}")
    st.caption("Chaque seuil est choisi sur la validation pour maximiser le F1 : au-dessus, le vol est classé « En retard ».")
