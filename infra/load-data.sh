#!/usr/bin/env bash
# Étape 4 – Mettre les données sur HDFS
# Prérequis : cluster démarré (docker compose up -d) et CSV présents dans ../data
# (dataset Kaggle : https://www.kaggle.com/datasets/usdot/flight-delays)
set -euo pipefail
cd "$(dirname "$0")"

DATA_DIR=../data
FILES=(flights.csv airlines.csv airports.csv)

for f in "${FILES[@]}"; do
  [ -f "$DATA_DIR/$f" ] || { echo "Fichier manquant : $DATA_DIR/$f (dézipper le dataset Kaggle dans data/ à la racine du repo)"; exit 1; }
done

echo "Attente des 2 DataNodes..."
until [ "$(docker exec namenode hdfs dfsadmin -report -live 2>/dev/null | grep -c '^Name:')" -ge 2 ]; do
  sleep 3
done

docker exec namenode hdfs dfs -mkdir -p /data/raw

# Envoi par stdin ("-") : pas de copie intermédiaire dans le container
for f in "${FILES[@]}"; do
  echo "-> /data/raw/$f"
  docker exec -i namenode hdfs dfs -put -f - "/data/raw/$f" < "$DATA_DIR/$f"
done

docker exec namenode hdfs dfs -ls -h /data/raw
