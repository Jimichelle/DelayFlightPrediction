# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ESGI school project (app and README in French, notebook in English): predicting US flight delays from the Kaggle USDOT 2015 dataset (`data/flights.csv`, 5,819,079 rows × 31 columns, stored in Git LFS). Pipeline: **CSV → HDFS (Docker) → PySpark notebook (EDA, cleaning, 4 models) → exported CSVs and models in `app/` → Streamlit app**.

A flight counts as delayed when `ARRIVAL_DELAY > 15` (the BTS definition). That gives `label` 1/0, with about 18% positives, so accuracy is misleading and AUC is the metric to watch.

## Commands

There are no tests, linter or build step.

```bash
uv sync                                   # Python 3.11 only (requires-python >=3.11,<3.12); requirements.txt mirrors this for pip

# HDFS cluster (1 NameNode + 2 DataNodes, Hadoop 3.4.1 on Java 11)
cd infra && docker compose up -d --build
./infra/load-data.sh                      # waits for the 2 DataNodes, then puts data/{flights,airlines,airports}.csv into hdfs:///data/raw/
# NameNode web UI: http://localhost:9870

uv run jupyter lab                        # notebooks/
uv run streamlit run app/app.py           # run from the repo root
```

## Gotchas when connecting Spark on the host to HDFS in Docker

- The host's `/etc/hosts` needs `127.0.0.1 namenode datanode1 datanode2`. The DataNodes advertise their hostnames, and each publishes its own block-transfer port (9866 and 9966), so the client must reach them by name.
- The Spark session needs `.config("spark.hadoop.dfs.client.use.datanode.hostname", "true")`. Notebooks read from `hdfs://localhost:9000/data/raw/...`.
- Java version: use Java 17 or 21 (`JAVA_HOME`), because the Hadoop client bundled with PySpark crashes on Java 23+ (`getSubject is not supported`). The notebook does not set `JAVA_HOME` itself.
- The NameNode formats itself only when its volume is empty (`infra/entrypoint.sh`). To reset HDFS, run `docker compose down -v`.
- `dfs.replication=2` and permissions are disabled (dev setup). DataNode-specific ports are passed as `-D` overrides in `docker-compose.yml`, not in `hdfs-site.xml`.

## Notebook (`notebooks/flight-delays.ipynb`, the only one)

- Pipeline: HDFS read → EDA → cleaning (5,231,110 flights) → features → 4 models → export. Runs end to end in about 5 min (`jupyter nbconvert --execute --inplace`, run from `notebooks/` because exports use `../app/...`).
- Spark: local mode, `spark.driver.memory=6g` (4g ran out of heap in the Random Forest), `spark.sql.shuffle.partitions=16`, a single cache on `df_clean` right after `dropDuplicates`.
- Models are trained on a `SAMPLE_FRACTION` sample (default 0.15) brought to pandas, one-hot rebuilt in NumPy, split 70/10/20 with a fixed seed. All 4 models use the same rows. The threshold maximizes F1 on validation.
  - Logistic regression and MLP: Keras, with `EarlyStopping`.
  - XGBoost: native `xgb.train` API, so scikit-learn is not needed. Needs `brew install libomp` on macOS.
  - Random Forest: Spark MLlib, takes raw inputs (6 numbers + 3 category indices).
- Leakage rule: only features known before departure (airline, airports, schedule, distance, date). `DEPARTURE_DELAY`, `TAXI_OUT` and the `*_DELAY` cause columns are post-departure or derived from the target.

## Streamlit app (`app/`)

Multipage app. `app/app.py` is the entry point: it holds `set_page_config` and `st.navigation`. Each page is a script in `app/pages/`:
- `predict.py`: pick origin, destination and airline (only real 2015 routes from `routes.csv`), date and hour, and get the delay probability from the 4 models (the default page)
- `explore.py`: delay rate and average delay by airline, hour and month, with filters, re-aggregated from `stats_detail.csv`
- `model.py`: test scores, precision/recall/F1 vs threshold, correlation matrix

Add a page by adding an `st.Page` to `app/app.py`; with `st.navigation`, Streamlit does not auto-discover `pages/`. Use `@st.cache_data` for data and `@st.cache_resource` for models. Every page shows an info message when an exported file is missing.

Files written by the notebook's export cell and read by the app:

| File | Content |
|---|---|
| `app/data/stats_airline.csv`, `stats_hour.csv` | global stats (columns from the original dashboard) |
| `app/data/stats_detail.csv` | `AIRLINE, MONTH, dep_hour, flights, delayed, sum_arr_delay` |
| `app/data/routes.csv` | `ORIGIN_AIRPORT, DESTINATION_AIRPORT, AIRLINE, flights, sched_time, distance` (medians) |
| `app/data/model_scores.csv` | `model, accuracy, f1, auc, threshold` |
| `app/data/threshold_curves.csv` | `model, threshold, precision, recall, f1` |
| `app/data/correlation.csv` | square Pearson matrix of the numeric features + `label` |
| `app/models/*.keras`, `xgboost.json`, `random_forest_spark/` | the 4 models |
| `app/models/preprocessing.json` | `num_cols`, `mean`, `std`, `cat_cols`, `cat_labels`, and `models`: `[{name, file, inputs, threshold, auc}]`, best AUC first |

`inputs` is `"onehot"` (standardized numbers + one-hot with an extra unknown slot, used by Keras and XGBoost) or `"raw"` (raw numbers + category indices, used by the Spark RF). `predict.py` `encode()` must stay in sync with the notebook's `to_features()` and `rf_assembler`. The predict page starts a local Spark session to load the RF, so Java is required. Airline and airport display names come from `data/airlines.csv` and `data/airports.csv`.
