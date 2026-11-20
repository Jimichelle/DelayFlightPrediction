# Explication du notebook `flight-delays.ipynb`

> Le notebook suit les étapes 4.1 à 4.6 du cours, puis exporte tout ce dont l'app Streamlit (4.7) a besoin.
> Il s'exécute de haut en bas en ~5 min (`jupyter nbconvert --execute --inplace`, lancé depuis `notebooks/`).

## Vue d'ensemble

| # | Section | Outil | Données |
|---|---|---|---|
| 1 | Démarrer Spark, lire depuis HDFS | Spark | 5 819 079 lignes |
| 2 | Parsing : types des colonnes | Spark | idem |
| 3 | EDA : valeurs manquantes, cible, corrélations | Spark | échantillon 5 % + dataset complet |
| 4 | Nettoyage | Spark | 5 231 110 lignes |
| 4.5 | Features : heure, label, encodage | Spark ML | idem |
| 5 | 4 modèles | pandas/NumPy, TensorFlow, XGBoost, Spark MLlib | échantillon 15 % = 785 169 lignes |
| 6 | Export pour Streamlit | Spark + pandas | stats sur tout, modèles sur l'échantillon |

**Le fil rouge** : Spark fait tout le travail sur les **5,2 M vols** (lecture, nettoyage, statistiques). Seul l'entraînement de 3 modèles se fait sur un **échantillon en mémoire**, parce que TensorFlow et XGBoost ne lisent pas les DataFrames Spark.

---

## 1. Démarrer Spark et lire le fichier depuis HDFS

```python
spark = (SparkSession.builder.appName("flight-delays")
         .config("spark.driver.memory", "6g")
         .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
         .config("spark.sql.shuffle.partitions", "16")
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         .getOrCreate())
```

| Réglage | Pourquoi |
|---|---|
| `spark.driver.memory=6g` | En mode local, tout tourne dans le driver. Avec 4 Go, le Random Forest manquait de mémoire. |
| `dfs.client.use.datanode.hostname=true` | Spark tourne hors de Docker : il doit joindre les DataNodes par leur **nom** (`datanode1`, `datanode2`), résolu via `/etc/hosts`. |
| `spark.sql.shuffle.partitions=16` | Par défaut 200. En local sur 8 cœurs, 200 petites tâches coûtent surtout en frais de gestion. |
| `arrow.pyspark.enabled=true` | Accélère fortement `toPandas()` : les données passent en format colonne (Arrow) au lieu d'être converties ligne par ligne. |

**Lecture :**
```python
raw = (spark.read.option("header", "true").option("inferSchema", "false")
       .option("delimiter", ",").csv("hdfs://localhost:9000/data/raw/flights.csv"))
n_raw = raw.count()   # 5 819 079 lignes, 31 colonnes
```
- `header=true` : la première ligne contient les noms de colonnes.
- `inferSchema=false` : tout est lu en texte. Sinon, Spark ferait une **lecture complète en plus** du fichier pour deviner les types.
- Le `count()` correspond au nombre de lignes annoncé sur Kaggle : c'est le point de contrôle de l'étape 4.1.

---

## 2. Parsing : donner le bon type aux colonnes

1. `raw.printSchema()` montre que toutes les colonnes sont des `string`.
2. Une cellule affiche la **première valeur non nulle** de chaque colonne (`F.first(c, ignorenulls=True)`), en une seule requête. On voit ainsi lesquelles sont numériques.
3. Résultat : **5 colonnes texte** (`AIRLINE`, `TAIL_NUMBER`, `ORIGIN_AIRPORT`, `DESTINATION_AIRPORT`, `CANCELLATION_REASON`) et **26 numériques**.
4. Cast des 26 colonnes numériques :
```python
for c in numerical_cols:
    df_parsed = df_parsed.withColumn(c, F.col(c).cast("double"))
```
Les noms de colonnes sont déjà courts et sans espaces, il n'y a donc rien à renommer.

---

## 3. Exploration (EDA)

### 3.0 Échantillon de 5 %
```python
eda_sample = df_parsed.sample(fraction=0.05, seed=42).cache()
```
~290 000 lignes suffisent pour explorer. `seed=42` rend l'échantillon reproductible. Ce `cache()` ne sert qu'à l'EDA.

### 3.1 Valeurs manquantes par colonne
```python
eda_sample.select([F.sum(F.isnull(c).cast("int")).alias(c) for c in eda_sample.columns])
```
Une seule requête Spark compte les valeurs nulles de **toutes** les colonnes à la fois (au lieu de 31 requêtes).
Constats :

- les variables connues avant le départ (mois, jour, compagnie, aéroports, heure, durée, distance) sont **complètes** ;
- les 5 colonnes `*_DELAY` (causes du retard) sont souvent vides. C'est normal : elles ne sont remplies que pour les vols **retardés**. Attention : le texte de la cellule 17 dit « annulés », c'est une erreur ;
- `ARRIVAL_DELAY` manque pour les vols annulés ou déroutés.

### 3.2 Valeurs distinctes des colonnes texte (dataset complet)
`F.countDistinct` sur les 5 colonnes, en une requête : 14 compagnies, 4 897 avions (`TAIL_NUMBER`), 629 / 628 aéroports (codes à 3 lettres **et** codes numériques), 4 raisons d'annulation.
Conséquence : les aéroports demandent un encodage soigné, et `TAIL_NUMBER` n'est pas utilisé.

### 3.3 Distribution de la cible
```python
valid_flights = (df_parsed.filter(F.col("ARRIVAL_DELAY").isNotNull())
                 .withColumn("is_delayed", (F.col("ARRIVAL_DELAY") > 15).cast("int")))
```
Un `groupBy("is_delayed").count()` sur tout le dataset, affiché en camembert : **82,1 % à l'heure, 17,9 % en retard**.
Conséquence : l'accuracy sera trompeuse, il faudra regarder l'AUC et le F1.

### 3.4 Corrélations avec la cible
`F.corr(colonne, is_delayed)` pour les 6 variables numériques, en une requête, affiché en barres horizontales.

- `SCHEDULED_DEPARTURE` ≈ **0,14** : plus le vol part tard dans la journée, plus il est souvent en retard (les retards s'accumulent).
- `MONTH` ≈ -0,04, les autres sont proches de 0.
- Conséquence : les liens linéaires sont faibles, il faut s'attendre à une AUC modeste.

### 3.5 Les 3 constats qui guident la suite
1. Les valeurs manquantes ne sont pas aléatoires : on retire les vols annulés ou déroutés.
2. `DEPARTURE_DELAY` et `TAXI_OUT` sont connus après le départ (**fuite de données**). Les classes sont déséquilibrées.
3. Les codes d'aéroport doivent être nettoyés.

---

## 4. Nettoyage (sur le dataset complet)

Chaque étape **compte les lignes avant et après**, comme le demande le cours.

### 4.1 Colonnes utiles et valeurs manquantes
```python
df_clean = (df_parsed.select(cols)
    .filter((F.col("CANCELLED") == 0) & (F.col("DIVERTED") == 0))
    .dropna(subset=[...]))
```
- `select(cols)` : on ne garde que 12 colonnes (variables d'avant le départ + `ARRIVAL_DELAY` + `CANCELLED`/`DIVERTED`). Spark ne lira pas les autres colonnes, et toutes les colonnes de fuite disparaissent dès cette étape.
- Retrait des vols annulés ou déroutés, puis des lignes incomplètes.
- **5 819 079 → 5 714 008** (-105 071).

### 4.2 Doublons et cache
```python
df_clean = df_clean.dropDuplicates().cache()
```
- **5 714 008 → 5 714 005** (-3).
- `dropDuplicates` provoque un **shuffle** (opération coûteuse). Toutes les étapes suivantes partent de `df_clean` : on le met donc en cache **une seule fois**, ici. Le `count()` qui suit remplit le cache.

### 4.3 Valeurs impossibles et outliers
D'abord **une seule requête** pour tout vérifier : nombre de codes d'aéroport numériques, min/max du retard, durée minimale, distance minimale.
Constats : 482 877 vols ont des codes d'aéroport numériques, le retard va de -87 à 1 971 min, et durée et distance sont toujours positives.
```python
df_clean = (df_clean
    .filter(~F.col("ORIGIN_AIRPORT").rlike("^[0-9]+$") & ~F.col("DESTINATION_AIRPORT").rlike("^[0-9]+$"))
    .filter((F.col("SCHEDULED_TIME") > 0) & (F.col("DISTANCE") > 0)
            & F.col("ARRIVAL_DELAY").between(-120, 1500)))
```
- `rlike("^[0-9]+$")` : expression régulière qui repère les codes composés uniquement de chiffres (octobre 2015 dans ce dataset). Impossible de les relier à `airports.csv`.
- Retard limité à [-120, 1500] min pour retirer les cas extrêmes.
- **5 714 005 → 5 231 110** (-482 895). Presque tout vient des codes numériques : le plafond du retard ne retire qu'une vingtaine de vols.

---

## 4.5 Feature engineering

### Heure de départ
```python
F.when(F.col("SCHEDULED_DEPARTURE") == 2400, 0).otherwise(...)
dep_hour = (sched_dep_clean / 100).cast("int")
```
`SCHEDULED_DEPARTURE` est au format `HHMM` (ex. `1745`). `2400` devient `0` pour ne pas obtenir une heure « 24 ». Diviser par 100 puis tronquer donne l'heure (17). On vérifie que min = 0 et max = 23.

### Cible
```python
label = F.when(F.col("ARRIVAL_DELAY") > 15, 1.0).otherwise(0.0)
```
Le nom `label` est celui qu'attendent les modèles Spark ML.

### Encodage des catégories
```python
indexer = StringIndexer(inputCols=cat_cols, outputCols=idx_cols, handleInvalid="keep")
indexer_model = indexer.fit(df_feat)
encoder = OneHotEncoder(inputCols=idx_cols, outputCols=ohe_cols)
```
- `StringIndexer` : chaque compagnie ou aéroport reçoit un numéro, **le plus fréquent en premier** (index 0). On le fait pour 3 colonnes à la fois.
- `handleInvalid="keep"` : une valeur inconnue reçoit un index supplémentaire au lieu de provoquer une erreur.
- `indexer_model` est **gardé** : la liste de ses catégories (`labelsArray`) est exportée pour l'app.
- `OneHotEncoder` : transforme l'index en vecteur de 0 et de 1 (ici pour montrer le résultat, le one-hot utilisé par les modèles est refait en NumPy à l'étape 5).
- Tailles : 14 compagnies, 322 aéroports de départ, 322 d'arrivée.

---

## 5. Entraînement des modèles

### 5.1 Échantillon, découpage et normalisation (la cellule la plus importante)

**a) Échantillon ramené en pandas**
```python
pdf = (df_indexed.sample(fraction=SAMPLE_FRACTION, seed=42)
       .select(*num_cols, *idx_cols, "label").toPandas())
```
- `SAMPLE_FRACTION` vaut 0,15 par défaut (modifiable avec une variable d'environnement) → **785 169 vols**.
- On ne ramène que **10 colonnes** : 6 numériques, 3 index, le label. Le one-hot complet (667 colonnes) ne passe jamais par Spark → pandas, sinon la mémoire ne suffit pas.

**b) Découpage 70 / 10 / 20**
```python
rng = np.random.default_rng(42)
perm = rng.permutation(len(pdf))
test_i, val_i, train_i = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
```
On mélange les numéros de ligne, puis on les découpe. On obtient **549 620 train / 78 516 validation / 157 033 test**. Les 4 modèles utilisent exactement ces lignes, donc leurs scores sont comparables.

**c) Standardisation calculée sur le train seulement**
```python
mean = pdf.iloc[train_i][num_cols].mean()
std  = pdf.iloc[train_i][num_cols].std().replace(0, 1)
```
Calculer la moyenne et l'écart-type sur tout l'échantillon ferait passer des informations du test dans l'entraînement (petite fuite). `replace(0, 1)` évite une division par zéro.

**d) `to_features()` : one-hot en NumPy**
```python
X = np.zeros((len(frame), n_features), dtype=np.float32)
X[:, :6] = (valeurs_numériques - mean) / std
X[rows, off + index] = 1.0      # pour chaque colonne catégorielle
```
- Une seule matrice remplie de zéros est créée d'avance : 6 colonnes numériques + 15 (compagnies) + 323 + 323 (aéroports) = **667 colonnes**.
- Pour chaque catégorie, on met un 1 à la bonne position (`offsets` = où commence chaque bloc).
- Le `+1` de chaque bloc correspond à la case « inconnu ».
- `float32` au lieu de `float64` : 2 fois moins de mémoire (les 3 matrices font déjà **2,09 Go**).

### 5.2 Fonctions d'évaluation

**`evaluate_tf(y_true, y_prob, name, threshold)`**
- transforme les probabilités en 0/1 avec le seuil ;
- calcule la **matrice de confusion** `[[TN, FP], [FN, TP]]` avec TensorFlow ;
- en déduit accuracy, précision, rappel, F1, ainsi que l'AUC (`tf.keras.metrics.AUC`) ;
- renvoie une ligne pour `model_scores.csv`.

**`best_threshold(y_true, y_prob)`**
- teste les seuils de **0,05 à 0,94** (pas de 0,01) et garde celui qui maximise le **F1** ;
- appelée **uniquement sur la validation**, jamais sur le test ;
- un `assert` vérifie qu'elle fonctionne sur un petit cas évident.

**`early_stop`**
```python
EarlyStopping(monitor="val_loss", patience=2, min_delta=1e-4, restore_best_weights=True)
```
Arrête l'entraînement si la perte sur la validation ne s'améliore pas pendant 2 époques, puis revient aux meilleurs poids.

### Le même schéma pour chaque modèle
1. entraîner sur le **train** (avec la validation pour l'early stopping) ;
2. choisir le seuil sur la **validation** (`best_threshold`) ;
3. prédire sur le **test** et évaluer (`evaluate_tf`) ;
4. ranger le modèle dans le dictionnaire `models` (pour l'export).

### 5.3 Régression logistique (TensorFlow)
```python
tf.keras.Sequential([Input(shape=(667,)), Dense(1, activation="sigmoid")])
compile(optimizer="adam", loss="binary_crossentropy")
fit(..., epochs=30, batch_size=1024, callbacks=[early_stop])
```
- Une seule couche avec une sigmoïde : c'est exactement une régression logistique (somme pondérée → probabilité entre 0 et 1).
- `binary_crossentropy` : la perte standard pour une classification oui/non.
- `adam` : l'optimiseur qui ajuste les poids.
- Lots de 1 024 lignes, 30 époques au maximum (l'early stopping arrête avant).
- `tf.random.set_seed(42)` : résultats reproductibles.
- **AUC 0,64**, seuil 0,17. C'est le modèle de référence.

### 5.4 MLP (réseau de neurones)
```python
Dense(128, relu) -> Dropout(0.2) -> Dense(64, relu) -> Dropout(0.2) -> Dense(1, sigmoid)
```
- 2 couches cachées : le réseau apprend des relations **non linéaires**.
- `relu` : l'activation des couches cachées.
- `Dropout(0.2)` : éteint au hasard 20 % des neurones à chaque étape d'entraînement, pour limiter le surapprentissage.
- Mêmes réglages d'entraînement que la régression logistique.
- **AUC 0,68**, seuil 0,19.

### 5.5 XGBoost
```python
dtrain = xgb.QuantileDMatrix(X_train, label=y_train)
params = {"objective": "binary:logistic", "eval_metric": "logloss", "eta": 0.1,
          "max_depth": 8, "tree_method": "hist", "seed": 42}
xgb.train(params, dtrain, num_boost_round=1000, evals=[(dval, "val")], early_stopping_rounds=20)
xgb_model = xgb_model[: xgb_model.best_iteration + 1]
```
- Même matrice que les modèles TensorFlow (la standardisation n'a pas d'effet sur les arbres).
- `binary:logistic` : sortie = probabilité de retard.
- `eta=0.1` : chaque nouvel arbre ne corrige qu'un peu (taux d'apprentissage).
- `max_depth=8` : profondeur maximale de chaque arbre.
- `tree_method="hist"` + `QuantileDMatrix` : les valeurs sont regroupées en intervalles avant de chercher les coupures, ce qui va beaucoup plus vite.
- Jusqu'à 1 000 arbres, mais **arrêt** si la log-loss de validation ne baisse plus pendant 20 arbres. Ici, le meilleur est l'arbre **393**.
- La dernière ligne **coupe** le modèle au meilleur arbre, pour que le modèle sauvegardé prédise comme celui qui a été évalué.
- **AUC 0,72**, seuil 0,23 : **le meilleur modèle**. Sur le test, il détecte 53 % des retards (rappel), et 36 % de ses alertes sont justes (précision).

### 5.6 Random Forest (Spark MLlib)
```python
rf_assembler = VectorAssembler(inputCols=num_cols + idx_cols, outputCol="features")
def to_spark(rows):
    return rf_assembler.transform(spark.createDataFrame(pdf.iloc[rows][rf_cols + ["label"]]))
rf = RandomForestClassifier(numTrees=100, maxDepth=10, maxBins=64,
                            subsamplingRate=0.5, featureSubsetStrategy="sqrt", seed=42)
```
- On repart des **mêmes lignes** (train/val/test) que les autres modèles, renvoyées vers Spark avec `createDataFrame`.
- `VectorAssembler` : réunit les 9 colonnes en une colonne vecteur `features`, le format attendu par Spark ML.
- Entrées **brutes** : 6 nombres + 3 index (9 valeurs au lieu de 667). Les arbres n'ont besoin ni de standardisation ni de one-hot.
- `numTrees=100` arbres. `maxDepth=10` (à 12, la mémoire manquait). `subsamplingRate=0.5` : chaque arbre voit la moitié du train. `featureSubsetStrategy="sqrt"` : environ 3 variables testées à chaque coupure.
- `rf_predict` garde le `label` **à côté** de la probabilité dans le même DataFrame, pour ne jamais mélanger l'ordre des lignes. `vector_to_array("probability")[1]` = probabilité de la classe « retard ».
- **AUC 0,66**, seuil 0,20.
- Limite : les index d'aéroport sont traités comme des nombres ordonnés (voir la fiche de révision, Q31).

---

## 6. Export pour Streamlit

L'app ne refait aucun calcul lourd : le notebook lui prépare tout.

### Fichiers de données (`app/data/`), calculés par Spark sur les 5,2 M vols
| Fichier | Contenu | Page |
|---|---|---|
| `stats_airline.csv` | par compagnie : vols, retard moyen, % de retards | (stats globales) |
| `stats_hour.csv` | par heure : vols, retard moyen | (stats globales) |
| `stats_detail.csv` | par (compagnie, mois, heure) : `flights`, `delayed`, `sum_arr_delay` | Explorer |
| `routes.csv` | par (départ, arrivée, compagnie) : nombre de vols, **médiane** de la durée et de la distance (`percentile_approx(..., 0.5)`) | Prédire |

`stats_detail.csv` contient des **sommes** et pas des pourcentages : l'app peut ainsi recalculer des taux exacts pour n'importe quel filtre.

### Fichiers calculés sur l'échantillon (pandas)
| Fichier | Contenu | Page |
|---|---|---|
| `model_scores.csv` | accuracy, F1, AUC, seuil de chaque modèle (test) | Modèle |
| `threshold_curves.csv` | précision, rappel, F1 pour chaque seuil de 0,02 à 0,80, sur le test | Modèle |
| `correlation.csv` | matrice de corrélation de Pearson (6 variables + `label`) | Modèle |

### Modèles (`app/models/`)
La boucle parcourt les modèles **du meilleur au moins bon (AUC)** et sauvegarde chacun dans son format :

- Random Forest → un **dossier** (`random_forest_spark/`), car Spark sauvegarde ses modèles en dossiers. Le chemin commence par `file://` pour forcer le disque local. Sans préfixe, Spark utilise le système de fichiers par défaut (`fs.defaultFS`) : ici c'est déjà le disque local, car le notebook ne le règle pas, mais avec la configuration du cluster ce serait HDFS et l'app ne trouverait plus le modèle ;
- XGBoost → `xgboost.json` ;
- Keras → `.keras`.

Le nom de fichier est fabriqué à partir du nom du modèle (minuscules, caractères spéciaux remplacés par `_`).

### `preprocessing.json`, le lien entre le notebook et l'app
```json
{ "models": [{"name", "file", "inputs", "threshold", "auc"}, ...],
  "num_cols": [...], "mean": [...], "std": [...],
  "cat_cols": [...], "cat_labels": [[...], [...], [...]] }
```
- `models` : trié par AUC, le premier est celui que l'app utilise pour la prédiction principale. `inputs` vaut `"onehot"` (Keras, XGBoost) ou `"raw"` (RF).
- `mean` / `std` : ceux du **train**, pour standardiser exactement pareil.
- `cat_labels` : l'ordre des catégories du `StringIndexer`, qui donne la position dans le one-hot. Une valeur inconnue va dans la dernière case.

La fonction `encode()` de l'app (`predict.py`) doit rester **identique** à `to_features()` et `rf_assembler` du notebook.

---

## Résumé des résultats

| Modèle | Bibliothèque | Entrées | AUC | F1 | Seuil |
|---|---|---|---|---|---|
| **XGBoost** | xgboost | 667 (one-hot) | **0,72** | 0,43 | 0,23 |
| MLP | TensorFlow | 667 (one-hot) | 0,68 | 0,38 | 0,19 |
| Random Forest | Spark MLlib | 9 (brutes) | 0,66 | 0,37 | 0,20 |
| Régression logistique | TensorFlow | 667 (one-hot) | 0,64 | 0,36 | 0,17 |

## Les bonnes pratiques visibles dans le code (à citer à l'oral)
1. **Une seule requête Spark pour plusieurs calculs** (valeurs manquantes, distincts, corrélations, vérifications) au lieu d'une requête par colonne.
2. **Un seul cache**, placé juste après le shuffle coûteux.
3. **Aucune fuite de données** : colonnes d'après le départ retirées dès l'étape 4.1, `mean`/`std` calculés sur le train, seuil choisi sur la validation.
4. **Reproductibilité** : `seed=42` partout (échantillons, découpage, TensorFlow, XGBoost, RF).
5. **Comparaison équitable** : les 4 modèles utilisent les mêmes lignes de train, de validation et de test.
6. **Mémoire maîtrisée** : 10 colonnes ramenées en pandas, one-hot fait en NumPy, `float32`.
