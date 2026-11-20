# Le notebook `flight-delays-bis.ipynb` expliqué pas à pas

> Pour chaque partie : **le but**, puis **le code étape par étape**, puis **ce qu'il faut retenir pour l'oral**.
> Le notebook se lance depuis `notebooks/` et s'exécute de haut en bas en ~5 min.

## Le parcours complet

| Partie | Ce qu'on fait | Outil | Lignes |
|---|---|---|---|
| 1 | Démarrer Spark, lire le CSV depuis HDFS | Spark | 5 819 079 |
| 2 | Donner le bon type à chaque colonne | Spark | 5 819 079 |
| 3 | Explorer les données (EDA) | Spark | échantillon 5 % + tout |
| 4 | Nettoyer | Spark | 5 231 110 |
| 4.5 | Créer les features et la cible | Spark ML | 5 231 110 |
| 5.1 | Préparer les données du MLP, de XGBoost et du RF | pandas / NumPy | 785 169 (15 %) |
| 5.2 | Fonctions d'évaluation | NumPy | |
| 5.3 | Régression logistique | **Spark ML Pipeline** | 261 005 (5 %) |
| 5.4 | Réseau de neurones (MLP) | TensorFlow | 785 169 |
| 5.5 | XGBoost | xgboost | 785 169 |
| 5.6 | Random Forest | **Spark MLlib** | 785 169 |
| 6 | Exporter pour Streamlit | Spark + pandas | |
| 7 | Application Streamlit | Streamlit | fichiers exportés |
| 8 | Lire la matrice de corrélation (EDA) | | |
| 9 | Questions : combinaisons absentes du dataset | | |
| | Légende : vocabulaire et abréviations | | |

**L'idée générale** : Spark fait tout le travail lourd sur les 5,2 M vols. Les modèles sont entraînés sur des échantillons, et le notebook exporte ensuite de petits fichiers que l'app Streamlit lit.

---

## Partie 1 : Démarrer Spark et lire depuis HDFS

**But** : obtenir un DataFrame Spark avec tout le fichier `flights.csv` stocké dans HDFS.

### Étape 1.1 : créer la session Spark
```python
spark = (SparkSession.builder.appName("flight-delays")
         .config("spark.driver.memory", "6g")
         .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
         .config("spark.sql.shuffle.partitions", "16")
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         .getOrCreate())
```
1. `SparkSession.builder` : point d'entrée de Spark. `appName` donne un nom au job.
2. `spark.driver.memory=6g` : Spark tourne en **mode local**, donc tout se passe dans le driver. 4 Go ne suffisaient pas au Random Forest.
3. `dfs.client.use.datanode.hostname=true` : Spark tourne sur la machine, HDFS dans Docker. Le NameNode renvoie les **noms** des DataNodes (`datanode1`, `datanode2`), que `/etc/hosts` fait pointer vers `127.0.0.1`.
4. `spark.sql.shuffle.partitions=16` : au lieu de 200 par défaut. En local sur 8 cœurs, 200 petites tâches coûtent surtout en frais de gestion.
5. `arrow...enabled=true` : accélère `toPandas()` en transférant les données par colonnes (format Arrow).
6. `getOrCreate()` : crée la session, ou réutilise celle qui existe déjà.

### Étape 1.2 : lire le CSV
```python
raw = (spark.read.option("header", "true").option("inferSchema", "false")
       .option("delimiter", ",").csv("hdfs://localhost:9000/data/raw/flights.csv"))
n_raw = raw.count()
```
1. `header=true` : la 1ʳᵉ ligne donne les noms de colonnes.
2. `inferSchema=false` : tout est lu en texte. Deviner les types obligerait Spark à **relire tout le fichier** une fois de plus.
3. `hdfs://localhost:9000/...` : adresse du NameNode (port RPC 9000).
4. `count()` est une **action** : c'est elle qui lance vraiment la lecture. Résultat : **5 819 079 lignes, 31 colonnes**, comme annoncé sur Kaggle.

**À retenir** : jusqu'au `count()`, Spark n'a rien lu. Il construit un plan (**évaluation paresseuse**).

---

## Partie 2 : Parsing (types des colonnes)

**But** : transformer les colonnes texte en nombres quand c'en sont.

### Étape 2.1 : inspecter
- `raw.printSchema()` : toutes les colonnes sont des `string`.
- `raw.limit(5).toPandas()` : aperçu des 5 premières lignes.
- `raw.select([F.first(c, ignorenulls=True) ...])` : la première valeur non vide de chaque colonne, en **une seule requête**. On voit lesquelles sont numériques.

### Étape 2.2 : caster
```python
for c in numerical_cols:
    df_parsed = df_parsed.withColumn(c, F.col(c).cast("double"))
```
1. `numerical_cols` liste les 26 colonnes numériques (les 5 autres restent du texte : compagnie, avion, aéroports, raison d'annulation).
2. `withColumn(c, ...)` remplace la colonne par sa version convertie en nombre décimal.
3. Rien n'est calculé ici : ce sont des **transformations**, pas des actions.

---

## Partie 3 : Exploration (EDA)

**But** : comprendre les données avant de les modifier, et en tirer 3 constats.

### Étape 3.1 : échantillon de 5 %
```python
eda_sample = df_parsed.sample(fraction=0.05, seed=42).cache()
```
~290 000 lignes suffisent pour explorer. `seed=42` rend le tirage reproductible, et `cache()` le garde en mémoire pour les requêtes suivantes.

### Étape 3.2 : valeurs manquantes
```python
eda_sample.select([F.sum(F.isnull(c).cast("int")).alias(c) for c in eda_sample.columns])
```
Pour chaque colonne : `isnull` donne vrai/faux, `cast("int")` le transforme en 1/0, et `sum` compte. Les 31 colonnes sont traitées en **une seule requête**.
Constats : les variables connues avant le départ sont complètes. `ARRIVAL_DELAY` manque pour les vols annulés ou déroutés. Les colonnes `*_DELAY` ne sont remplies que pour les vols **retardés**.

### Étape 3.3 : valeurs distinctes (dataset complet)
`F.countDistinct` sur les 5 colonnes texte : 14 compagnies, 4 897 avions, 629 / 628 aéroports (dont des codes numériques).

### Étape 3.4 : distribution de la cible
```python
valid_flights = (df_parsed.filter(F.col("ARRIVAL_DELAY").isNotNull())
                 .withColumn("is_delayed", (F.col("ARRIVAL_DELAY") > 15).cast("int")))
counts = valid_flights.groupBy("is_delayed").count().toPandas()
```
1. On garde les vols qui ont un retard connu.
2. `is_delayed` = 1 si le retard dépasse 15 min (définition officielle du BTS).
3. `groupBy().count()` compte chaque classe, sur tout le dataset. Le petit résultat passe en pandas pour le camembert.

Résultat : **82,1 % à l'heure, 17,9 % en retard**. Les classes sont déséquilibrées : l'accuracy sera trompeuse.

### Étape 3.5 : corrélations
`F.corr(colonne, is_delayed)` pour les 6 variables numériques, en une requête. La plus forte est `SCHEDULED_DEPARTURE` (≈ 0,14) : plus le vol est tard dans la journée, plus il risque d'être en retard. Les liens sont faibles, il faut donc s'attendre à une AUC modeste.

### Les 3 constats
1. Les valeurs manquantes ne sont pas aléatoires : on retire les vols annulés ou déroutés.
2. `DEPARTURE_DELAY` et `TAXI_OUT` ne sont connus qu'après le départ (**fuite de données**). Les classes sont déséquilibrées.
3. Les codes d'aéroport doivent être nettoyés.

---

## Partie 4 : Nettoyage

**But** : un dataset sans valeur manquante ni valeur aberrante, en comptant les lignes à chaque étape.

### Étape 4.1 : colonnes utiles et lignes incomplètes
```python
df_clean = (df_parsed.select(cols)
    .filter((F.col("CANCELLED") == 0) & (F.col("DIVERTED") == 0))
    .dropna(subset=[...]))
```
1. `select(cols)` : on ne garde que 12 colonnes. Toutes les colonnes de fuite (`DEPARTURE_DELAY`, `TAXI_OUT`, `*_DELAY`…) disparaissent dès cette étape.
2. `filter` : on retire les vols annulés ou déroutés, qui n'ont pas de retard à l'arrivée.
3. `dropna` : on retire les lignes où il manque une valeur utile.
4. **5 819 079 → 5 714 008**.

### Étape 4.2 : doublons et cache
```python
df_clean = df_clean.dropDuplicates().cache()
after_dedup = df_clean.count()
```
1. `dropDuplicates` retire les lignes identiques : **5 714 008 → 5 714 005**.
2. Cette opération fait un **shuffle** : les données sont redistribuées entre partitions, ce qui coûte cher.
3. `cache()` : toutes les étapes suivantes partent de `df_clean`. Sans cache, chaque action relirait HDFS et referait le shuffle.
4. Le `count()` remplit le cache, qui est lui-même paresseux.

### Étape 4.3 : valeurs impossibles et outliers
```python
df_clean = (df_clean
    .filter(~F.col("ORIGIN_AIRPORT").rlike("^[0-9]+$") & ~F.col("DESTINATION_AIRPORT").rlike("^[0-9]+$"))
    .filter((F.col("SCHEDULED_TIME") > 0) & (F.col("DISTANCE") > 0)
            & F.col("ARRIVAL_DELAY").between(-120, 1500)))
```
1. Avant de filtrer, une seule requête vérifie tout : codes numériques, retard minimal et maximal, durée et distance minimales.
2. `rlike("^[0-9]+$")` repère les codes d'aéroport composés uniquement de chiffres. `~` signifie « non », on les retire donc. Ce sont des vols d'octobre 2015 qu'on ne peut pas relier à `airports.csv`.
3. Durée et distance doivent être positives, et le retard compris entre -120 et 1 500 min.
4. **5 714 005 → 5 231 110**. Les codes numériques représentent presque tout (482 877 lignes), les outliers seulement une vingtaine.

---

## Partie 4.5 : Features et cible

**But** : transformer les colonnes nettoyées en données qu'un modèle peut apprendre.

### Étape 4.5.1 : heure de départ
```python
df_feat = df_clean.withColumn("sched_dep_clean",
    F.when(F.col("SCHEDULED_DEPARTURE") == 2400, 0).otherwise(F.col("SCHEDULED_DEPARTURE").cast("int")))
df_feat = df_feat.withColumn("dep_hour", (F.col("sched_dep_clean") / 100).cast("int"))
```
1. `SCHEDULED_DEPARTURE` est au format `HHMM` (ex. 1745).
2. `2400` (minuit) devient `0`, sinon on obtiendrait une heure « 24 ».
3. Diviser par 100 puis tronquer en entier donne l'heure : 1745 / 100 = 17,45, qui devient 17.
4. On vérifie que `dep_hour` va bien de 0 à 23.

### Étape 4.5.2 : la cible `label`
```python
df_feat = df_feat.withColumn("label", F.when(F.col("ARRIVAL_DELAY") > 15, 1.0).otherwise(0.0))
```
On utilise le nom `label` parce que c'est celui qu'attendent par défaut les modèles Spark ML.

### Étape 4.5.3 : encoder les catégories
```python
indexer = StringIndexer(inputCols=cat_cols, outputCols=idx_cols, handleInvalid="keep")
indexer_model = indexer.fit(df_feat)
df_indexed = indexer_model.transform(df_feat)
df_encoded = OneHotEncoder(inputCols=idx_cols, outputCols=ohe_cols).fit(df_indexed).transform(df_indexed)
```
1. `StringIndexer` donne un numéro à chaque compagnie ou aéroport, **le plus fréquent en premier** (0).
2. `fit` apprend la liste des valeurs, puis `transform` ajoute les colonnes d'index.
3. `handleInvalid="keep"` : une valeur inconnue reçoit un index de plus, au lieu de provoquer une erreur.
4. `indexer_model` est gardé : sa liste de catégories est exportée pour l'app.
5. `OneHotEncoder` transforme chaque index en vecteur de 0 et de 1. Ici, c'est surtout pour montrer le résultat.
6. Tailles : 14 compagnies, 322 aéroports de départ, 322 d'arrivée.

---

## Partie 5.1 : Préparer les données du MLP, de XGBoost et du Random Forest

**But** : un échantillon en mémoire, découpé en train / validation / test, avec le one-hot construit en NumPy.

### Étape 5.1.1 : ramener un échantillon en pandas
```python
pdf = (df_indexed.sample(fraction=SAMPLE_FRACTION, seed=42)
       .select(*num_cols, *idx_cols, "label").toPandas())
```
1. 15 % des vols (**785 169**). La fraction se règle avec la variable d'environnement `SAMPLE_FRACTION`.
2. On ne ramène que **10 colonnes** : 6 numériques, 3 index, le label. Ramener le one-hot complet (667 colonnes) saturait la mémoire.

### Étape 5.1.2 : découper 70 / 10 / 20
```python
rng = np.random.default_rng(42)
perm = rng.permutation(len(pdf))
test_i, val_i, train_i = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
```
1. `permutation` mélange les numéros de ligne (graine 42, donc toujours le même mélange).
2. Les premiers 20 % vont au test, les 10 % suivants à la validation, le reste au train.
3. On obtient **549 620 train / 78 516 validation / 157 033 test**. Le MLP, XGBoost et le RF utilisent exactement ces lignes.

### Étape 5.1.3 : standardisation calculée sur le train
```python
mean = pdf.iloc[train_i][num_cols].mean()
std  = pdf.iloc[train_i][num_cols].std().replace(0, 1)
```
On calcule la moyenne et l'écart-type **uniquement sur le train**, pour qu'aucune information du test n'entre dans la préparation (fuite). `replace(0, 1)` évite une division par zéro.

### Étape 5.1.4 : construire le one-hot (`to_features`)
```python
X = np.zeros((len(frame), n_features), dtype=np.float32)
X[:, :6] = (valeurs_numériques - mean) / std
X[rows, offset + index] = 1.0
```
1. Une matrice de zéros est créée d'avance : 6 + 15 + 323 + 323 = **667 colonnes**. Le `+1` de chaque bloc est la case « inconnu ».
2. Les 6 premières colonnes reçoivent les nombres standardisés.
3. Pour chaque catégorie, on met un 1 à la bonne position. `offset` indique où commence le bloc de la catégorie.
4. `float32` divise la mémoire par 2 : les matrices font déjà 2,09 Go.

---

## Partie 5.2 : Fonctions d'évaluation (NumPy, sans TensorFlow)

**But** : mesurer tous les modèles de la même façon.

### Étape 5.2.1 : `auc_score`, l'AUC calculée à la main
```python
ranks = pd.Series(y_prob).rank().to_numpy()
auc = (ranks[y_true].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
```
1. On classe toutes les probabilités du plus petit au plus grand (`rank`). En cas d'égalité, on prend le rang moyen.
2. On additionne les rangs des vols vraiment retardés.
3. La formule transforme cette somme en **probabilité qu'un vol retardé ait un score plus haut qu'un vol à l'heure**, c'est-à-dire l'AUC.
4. Un `assert` vérifie la formule sur un petit exemple dont on connaît la réponse (0,75).

### Étape 5.2.2 : `evaluate`
1. Transforme chaque probabilité en 0 ou 1 avec le seuil (`y_prob >= threshold`).
2. Compte TP, FP, FN et TN, ce qui donne la **matrice de confusion**.
3. En déduit l'accuracy, la précision, le rappel, le F1 et l'AUC.
4. Affiche le tout et renvoie une ligne pour `model_scores.csv`.

### Étape 5.2.3 : `best_threshold`
1. Teste chaque seuil de 0,05 à 0,94, par pas de 0,01.
2. Garde celui qui donne le **meilleur F1**.
3. Elle est toujours appelée sur la **validation**, jamais sur le test.

### Étape 5.2.4 : préparer les listes
`results = []` reçoit le score de chaque modèle, et `models = {}` les modèles à exporter.

---

## Partie 5.3 : Régression logistique avec un Pipeline Spark ML

**But** : un modèle entièrement `pyspark.ml`, avec le découpage demandé par le cours.

### Étape 5.3.1 : échantillon et découpage 80/20 dans Spark
```python
spark_ml_population = df_feat.select(*(spark_ml_cat_cols + spark_ml_num_cols + ["label"]))
spark_ml_sample = spark_ml_population.sample(False, 0.05, 42).persist(StorageLevel.MEMORY_ONLY)
spark_ml_train, spark_ml_test = spark_ml_sample.randomSplit([0.8, 0.2], seed=42)
```
1. `select` : 3 colonnes texte (compagnie, départ, arrivée), 6 numériques et le label.
2. `sample(False, 0.05, 42)` : 5 % des vols, sans remise, graine 42, soit **261 005 vols**.
3. `persist(MEMORY_ONLY)` : comme `cache()`, en précisant qu'on stocke en mémoire uniquement.
4. `randomSplit([0.8, 0.2], seed=42)` : exactement la consigne de la slide 57. On obtient **209 228 train / 51 777 test**.
5. Les `assert` vérifient que train + test = échantillon et qu'aucun des deux n'est vide.
6. `groupBy("label").count()` affiche la répartition des classes : environ 18,5 % de retards dans chaque partie.

### Étape 5.3.2 : le Pipeline (`spark_ml_pipeline`)
```python
Pipeline(stages=[indexer, encoder, assembler, classifier])
```
Un Pipeline enchaîne les étapes : un seul `fit()` les apprend toutes, dans l'ordre.

1. **`StringIndexer`** : les 3 colonnes texte deviennent des index (`handleInvalid="keep"` pour les valeurs inconnues).
2. **`OneHotEncoder`** : les index deviennent des vecteurs de 0 et de 1.
3. **`VectorAssembler`** : les 6 nombres et les 3 vecteurs sont réunis en **une seule colonne `spark_ml_features`**, de **663** dimensions. C'est le format qu'exige Spark ML.
4. **`LogisticRegression`** :
   - `maxIter=30` : au plus 30 itérations de l'optimiseur ;
   - `regParam=0.05` : **régularisation**, qui pénalise les poids trop grands pour limiter le surapprentissage ;
   - `standardization=True` : Spark standardise les variables lui-même, sans `StandardScaler`.

**Avantage du Pipeline** : les règles d'encodage sont apprises sur le train seulement, puis appliquées telles quelles au test. Il n'y a pas de fuite possible.

### Étape 5.3.3 : la fonction de métriques (`spark_ml_metrics`)
1. `MulticlassClassificationEvaluator(metricName="accuracy")` calcule l'accuracy.
2. `MulticlassClassificationEvaluator(metricName="f1")` calcule le **F1 pondéré** : la moyenne des F1 des deux classes, pondérée par leur taille. Avec 82 % de vols à l'heure, il est surtout tiré par la classe « à l'heure ».
3. `BinaryClassificationEvaluator(metricName="areaUnderROC")` calcule l'AUC à partir du score `spark_ml_score`.
4. Un `groupBy("label", "prediction").count()` donne la matrice de confusion, d'où la précision, le rappel et le **F1 de la classe retard**.

### Étape 5.3.4 : la baseline majoritaire
1. On cherche la classe la plus fréquente du train : 0, « à l'heure ».
2. On crée un faux modèle qui prédit toujours 0.
3. Résultat : accuracy de **81,3 %**, AUC de **0,5**, aucun retard détecté. C'est le niveau à battre, et il montre bien que l'accuracy seule ne veut rien dire ici.

### Étape 5.3.5 : entraîner et évaluer
```python
spark_ml_model = spark_ml_pipeline().fit(spark_ml_train)
spark_ml_predictions = spark_ml_model.transform(spark_ml_test)
spark_ml_evaluation = spark_ml_predictions.select("label", "prediction",
    vector_to_array("probability")[1].alias("spark_ml_score")).persist(...)
```
1. `time.perf_counter()` avant et après le `fit` mesure le temps d'entraînement : environ **4 s**.
2. `transform(test)` ajoute les colonnes `probability` (vecteur [P(à l'heure), P(retard)]) et `prediction`.
3. `vector_to_array("probability")[1]` extrait la probabilité de retard.
4. Le `try / finally` libère la mémoire (`unpersist`), même en cas d'erreur.
5. `spark_ml_model.stages[-1].numFeatures` : la dernière étape du Pipeline est la régression, qui a 663 entrées.

### Étape 5.3.6 : comparaison et liaison avec le reste
1. Un tableau compare la baseline et la régression logistique.
2. La cellule de liaison ajoute le modèle à `results`, avec le **F1 de la classe retard** (comparable aux autres modèles, contrairement au F1 pondéré). Elle récupère aussi ses probabilités sur le test pour les courbes de seuil.

### Le résultat, et sa limite
| | Accuracy | AUC | Rappel retard | F1 retard |
|---|---|---|---|---|
| Baseline | 81,3 % | 0,500 | 0 % | 0 |
| Régression logistique Spark | 81,3 % | **0,635** | **0 %** | **0** |

L'AUC de 0,635 montre que le modèle **sait classer** les vols. Mais `prediction` utilise le seuil par défaut de Spark, **0,5**, et avec 18 % de retards aucune probabilité ne le dépasse. Le modèle répond donc toujours « à l'heure ». Correction possible : choisir le seuil qui maximise le F1 sur les prédictions du train.

---

## Partie 5.4 : Réseau de neurones (MLP, TensorFlow)

**But** : un modèle non linéaire, entraîné sur les matrices one-hot de la partie 5.1.

```python
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2,
                                              min_delta=1e-4, restore_best_weights=True)
mlp_tf = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(667,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
mlp_tf.compile(optimizer="adam", loss="binary_crossentropy")
mlp_tf.fit(X_train, y_train, epochs=30, batch_size=1024,
           validation_data=(X_val, y_val), callbacks=[early_stop])
```
1. **Entrée** : les 667 colonnes (6 nombres standardisés + one-hot).
2. **2 couches cachées** de 128 puis 64 neurones, activation `relu` : elles permettent d'apprendre des relations **non linéaires** (par exemple, un effet de l'heure qui n'est pas une ligne droite).
3. **`Dropout(0.2)`** : pendant l'entraînement, 20 % des neurones sont éteints au hasard à chaque lot, ce qui limite le surapprentissage. Il est **désactivé** au moment de prédire.
4. **Sortie** : un neurone `sigmoid`, qui donne une probabilité de retard entre 0 et 1.
5. `adam` ajuste les poids, et `binary_crossentropy` est la perte standard d'une classification oui/non.
6. Lots de 1 024 vols, au plus 30 époques. **Early stopping** : si la perte de validation ne baisse plus pendant 2 époques, on arrête et on reprend les meilleurs poids.
7. `tf.random.set_seed(42)` rend les résultats presque reproductibles (TensorFlow ne l'est pas totalement).
8. Seuil choisi sur la validation (0,18), puis évaluation sur le test avec `evaluate`.

**Résultat** : AUC **0,679**, F1 0,385, précision 28 %, rappel 62 %. C'est le **2ᵉ meilleur modèle**.

---

## Partie 5.5 : XGBoost

**But** : le modèle le plus performant, avec des arbres boostés.

### Comment XGBoost fonctionne

**L'idée en une phrase** : XGBoost construit des centaines de petits arbres de décision **les uns après les autres**, et chaque nouvel arbre sert à **corriger les erreurs** de tous les précédents. C'est le **boosting** (*eXtreme Gradient Boosting*).

**1. Un arbre de décision, c'est quoi ?**
Une suite de questions oui/non sur les variables, qui mène à une feuille contenant un nombre :
```
heure de départ >= 15 ?
  oui -> aéroport de départ = ORD ?
           oui -> +0,30
           non -> +0,12
  non -> mois = juin ou juillet ?
           ...
```
Avec `max_depth=8`, un arbre pose au plus 8 questions à la suite. Il peut donc combiner jusqu'à 8 conditions, par exemple « compagnie = B6 **et** départ = JFK **et** heure >= 17 ». C'est ce qu'un modèle linéaire ne sait pas faire.

**2. Le score s'additionne arbre après arbre**
1. **Départ** : tous les vols reçoivent le même score de base, qui correspond au taux moyen de retard (~18 %).
2. **Arbre 1** : on regarde l'erreur de chaque vol. Pour un vol en retard prédit à 18 %, l'erreur est grande et positive ; pour un vol à l'heure, elle est petite et négative. Techniquement, c'est le **gradient** de la log-loss : `probabilité prédite - vraie valeur`. L'arbre 1 apprend à **prédire ces erreurs** à partir des variables.
3. On ajoute la sortie de l'arbre au score, **multipliée par `eta=0.1`** (le taux d'apprentissage). On ne corrige que 10 % de l'erreur à chaque fois : c'est plus lent, mais plus sûr contre le surapprentissage.
4. **Arbre 2** : il apprend les erreurs qui **restent** après l'arbre 1, et ainsi de suite.
5. **Prédiction finale** : on additionne le score de base et les sorties des 394 arbres, puis on passe le total dans une **sigmoïde**, qui le ramène entre 0 et 1. On obtient la probabilité de retard.

**3. Comment un arbre choisit ses questions**
- Pour chaque variable, XGBoost teste des seuils de coupure et garde celui qui **réduit le plus l'erreur** (le *gain*).
- Avec `tree_method="hist"`, les valeurs de chaque variable sont d'abord regroupées en **intervalles** (256 au maximum). On teste donc 256 coupures au lieu d'une par valeur différente, ce qui va beaucoup plus vite sur 550 000 vols.
- Avec le one-hot, une question porte sur **une seule catégorie** : « départ = ORD ? » (la colonne vaut 1 ou 0).
- Une **régularisation** est incluse dans le calcul (paramètre `lambda`, 1 par défaut) : elle évite de donner un poids énorme à une feuille qui ne contient que quelques vols.

**4. Quand s'arrêter**
- Au plus 1 000 arbres (`num_boost_round`).
- Après chaque arbre, on mesure la **log-loss sur la validation**. Si elle ne baisse plus pendant **20 arbres** (`early_stopping_rounds`), on arrête.
- Ici, le meilleur est l'arbre **393**. Les ~20 arbres suivants sont jetés par `xgb_model[: best_iteration + 1]`.

**5. Pourquoi XGBoost gagne ici**
- Le retard dépend de **combinaisons** (compagnie × aéroport × heure), que les arbres captent naturellement.
- L'effet de l'heure ou du mois n'est **pas linéaire**, et les arbres coupent où il faut.
- La standardisation ne change rien pour des arbres : seuls les seuils de coupure comptent.
- Comparé au Random Forest : la forêt fait **voter** des arbres indépendants et profonds, alors que XGBoost **empile** des corrections. Sur ces données, empiler des corrections donne une meilleure AUC (0,724 contre 0,665).

**6. Dans l'app**
L'app recharge `xgboost.json`, encode la demande en vecteur one-hot de 667 colonnes (comme `to_features()`), puis `model.predict(xgb.DMatrix(x))` additionne les sorties des 394 arbres et renvoie la probabilité.

### Le code étape par étape

```python
dtrain = xgb.QuantileDMatrix(X_train, label=y_train)
dval = xgb.QuantileDMatrix(X_val, label=y_val, ref=dtrain)
xgb_model = xgb.train(params, dtrain, num_boost_round=1000,
                      evals=[(dval, "val")], early_stopping_rounds=20)
xgb_model = xgb_model[: xgb_model.best_iteration + 1]
```
1. `QuantileDMatrix` prépare les mêmes matrices que le MLP (partie 5.1) au format de XGBoost, avec les valeurs regroupées en intervalles (`tree_method="hist"`, plus rapide).
2. Paramètres : `binary:logistic` (sortie = probabilité), `eta=0.1` (chaque arbre corrige un peu), `max_depth=8`, `seed=42`.
3. Jusqu'à 1 000 arbres, chacun corrigeant les erreurs des précédents (**boosting**).
4. **Early stopping** : si la log-loss de validation ne baisse plus pendant 20 arbres, on arrête. Le meilleur est l'arbre **393**.
5. Le découpage `[: best_iteration + 1]` garde les arbres 0 à 393 et jette les suivants, pour que le modèle sauvegardé soit celui qui a été évalué.
6. Seuil choisi sur la validation (0,23), puis évaluation sur le test.

**Résultat** : AUC **0,724**, F1 0,43, précision 36 %, rappel 53 %. C'est le **meilleur modèle**.

---

## Partie 5.6 : Random Forest (Spark MLlib)

**But** : un 2ᵉ modèle Spark, entraîné sur les mêmes lignes que XGBoost pour une comparaison équitable.

```python
rf_assembler = VectorAssembler(inputCols=num_cols + idx_cols, outputCol="features")
def to_spark(rows):
    return rf_assembler.transform(spark.createDataFrame(pdf.iloc[rows][rf_cols + ["label"]]))
rf = RandomForestClassifier(numTrees=100, maxDepth=10, maxBins=64,
                            subsamplingRate=0.5, featureSubsetStrategy="sqrt", seed=42)
rf_model = rf.fit(to_spark(train_i))
```
1. `to_spark(rows)` renvoie les lignes choisies (train, validation ou test) de pandas vers Spark, puis `VectorAssembler` crée la colonne `features`.
2. Entrées **brutes** : 6 nombres et 3 index, soit 9 valeurs. Les arbres n'ont besoin ni de standardisation ni de one-hot.
3. 100 arbres de profondeur 10 au maximum (à 12, la mémoire ne suffisait plus).
4. `subsamplingRate=0.5` : chaque arbre voit une moitié aléatoire des vols. `featureSubsetStrategy="sqrt"` : environ 3 variables testées à chaque coupure. Ces deux réglages rendent les arbres **différents**, ce qui fait la force de la forêt.
5. `rf_predict` garde le `label` à côté de la probabilité, pour ne jamais mélanger l'ordre des lignes.
6. Seuil choisi sur la validation (0,20), puis évaluation sur le test.

**Résultat** : AUC **0,665**, F1 0,37, rappel 59 %.

---

## Partie 6 : Export pour Streamlit

**But** : écrire de petits fichiers pour que l'app n'ait jamais besoin de Spark ni de HDFS pour afficher les statistiques.

### Étape 6.1 : statistiques, calculées par Spark sur les 5,2 M vols
| Fichier | Calcul | Utilisé par |
|---|---|---|
| `stats_airline.csv` | `groupBy("AIRLINE")` : vols, retard moyen, % de retards | stats globales |
| `stats_hour.csv` | `groupBy("dep_hour")` : vols, retard moyen | stats globales |
| `routes.csv` | `groupBy(départ, arrivée, compagnie)` : vols, **médiane** de la durée et de la distance (`percentile_approx(..., 0.5)`) | page Prédire |
| `stats_detail.csv` | `groupBy(compagnie, mois, heure)` : vols, retards, **somme** des minutes de retard | page Explorer |

Le fichier détaillé contient des **sommes** : l'app peut ainsi recalculer des taux exacts pour n'importe quel filtre.

### Étape 6.2 : fichiers sur les modèles
1. `threshold_curves.csv` : pour chaque modèle et chaque seuil de 0,02 à 0,80, la précision, le rappel et le F1 sur le test.
2. `correlation.csv` : matrice de corrélation des 6 variables et du label, sur l'échantillon de 15 %.
3. `model_scores.csv` : accuracy, F1, AUC et seuil de chaque modèle.

### Étape 6.3 : sauvegarde des modèles
```python
for row in scores.sort_values("auc", ascending=False).itertuples():
    if row.model not in models:
        continue
```
1. Les modèles sont parcourus du meilleur au moins bon (AUC).
2. La régression logistique Spark est **ignorée** : l'app ne sait pas lui préparer une entrée, qui doit passer par un DataFrame Spark avec les colonnes texte.
3. Le RF est sauvegardé en **dossier** (`random_forest_spark/`), avec un chemin `file://` qui force le disque local.
4. XGBoost est sauvegardé en `xgboost.json`, et le MLP en `neural_network_mlp_tf.keras`.

### Étape 6.4 : `preprocessing.json`
Il contient tout ce dont l'app a besoin pour encoder une demande **exactement** comme le notebook :

- `models` : nom, fichier, type d'entrée (`onehot` ou `raw`), seuil et AUC de chaque modèle, le meilleur en premier ;
- `num_cols`, `mean`, `std` : la standardisation du train ;
- `cat_cols`, `cat_labels` : l'ordre des catégories du `StringIndexer`.

---

## Partie 7 : L'application Streamlit (`app/`)

**But** : un dashboard qui montre les données et le modèle, sans relancer le notebook. L'app ne fait **aucun calcul lourd** : elle lit les fichiers de la partie 6 et ne se connecte jamais à HDFS.

```
app/
  app.py          point d'entrée : menu des 3 pages
  common.py       chemins et fonctions partagés
  pages/
    predict.py    Prédire un retard (page par défaut)
    explore.py    Explorer les retards
    model.py      Le modèle
  data/  models/  fichiers exportés par le notebook
```
Lancement depuis la racine du repo : `uv run streamlit run app/app.py`, puis ouvrir `localhost:8501`.

### Étape 7.1 : le principe de Streamlit à connaître
1. À **chaque interaction** (un clic, un menu, un curseur), Streamlit **relance tout le script de la page** de haut en bas.
2. Sans précaution, il relirait donc les CSV et rechargerait les modèles à chaque clic.
3. Deux décorateurs l'évitent :
   - `@st.cache_data` pour les **données** (CSV, JSON) : lues une fois, puis gardées en mémoire ;
   - `@st.cache_resource` pour les **ressources** (modèles, session Spark) : chargées une seule fois et partagées.

### Étape 7.2 : `app.py`, le point d'entrée
```python
st.set_page_config(page_title="Retards de vols US 2015", layout="wide")
nav = st.navigation([
    st.Page("pages/predict.py", title="Prédire un retard", default=True),
    st.Page("pages/explore.py", title="Explorer les retards"),
    st.Page("pages/model.py", title="Le modèle"),
])
nav.run()
```
1. `set_page_config` : titre de l’onglet et page en pleine largeur.
2. `st.navigation` déclare les 3 pages et construit le menu. `default=True` désigne la page d'accueil.
3. `nav.run()` exécute la page choisie.
4. Avec `st.navigation`, Streamlit ne découvre **pas** tout seul le dossier `pages/` : une nouvelle page doit être ajoutée dans cette liste.

### Étape 7.3 : `common.py`, ce que les pages partagent
```python
ROOT = Path(__file__).parents[1]
DATA, MODELS, RAW = ROOT / "app" / "data", ROOT / "app" / "models", ROOT / "data"

def require(*paths):
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        st.info(f"Fichiers manquants : {', '.join(missing)}. {RERUN}")
        st.stop()
```
1. Les chemins sont calculés une seule fois, à partir de l'emplacement du fichier. L'app fonctionne donc quel que soit le dossier d'où on la lance.
2. `require(...)` vérifie que les fichiers exportés existent. Sinon, la page affiche « Fichiers manquants… Relance l'export du notebook » et `st.stop()` l'arrête proprement, au lieu de planter.
3. `airline_names()` lit `data/airlines.csv` pour afficher « AA – American Airlines Inc. » au lieu de « AA ». Elle est en cache.

### Étape 7.4 : `predict.py`, prédire un retard

**a) Vérifier les fichiers.** `require(PREPROCESSING_JSON, ROUTES_CSV)`, puis une vérification que l'export n'est pas d'une ancienne version, puis `require(...)` sur chaque modèle listé dans `preprocessing.json`.

**b) Reconnaître le type de chaque modèle (`kind`)**
```python
def kind(file):
    if file.endswith("_spark"): return "spark"     # dossier Spark MLlib
    if file.endswith(".json"):  return "xgboost"
    return "keras"
```
Le type se déduit du nom de fichier donné par l'export. Trois fonctions s'appuient dessus avec un `match` :

- `load_models` : charge chaque modèle. Pour Spark, elle démarre une petite session locale (`local[1]`, sans interface web). Les imports lents (TensorFlow, XGBoost, Spark) sont faits seulement pour les types de modèles présents ;
- `n_inputs` : nombre d'entrées attendues par le modèle ;
- `delay_proba` : probabilité de retard (`predictProbability` pour Spark, `predict` pour XGBoost).

**c) Le formulaire, limité aux vols réels**
1. **Départ** : tous les aéroports d'origine de `routes.csv`.
2. **Arrivée** : seulement les destinations desservies depuis ce départ.
3. **Compagnie** : seulement celles qui volent sur cette route, triées par nombre de vols.
4. **Date** et **heure** : choisies par l'utilisateur.
5. La **durée** et la **distance** ne sont pas demandées : elles viennent de `routes.csv` (médianes de la route pour cette compagnie). Elles sont affichées sous le formulaire.

**d) Encoder la demande (`encode`)**, exactement comme le notebook, à partir de `preprocessing.json` :

- mode `onehot` (MLP et XGBoost) : les 6 nombres standardisés avec `mean` et `std` **du train**, puis un vecteur one-hot par catégorie, avec une case « inconnu » ;
- mode `raw` (Random Forest) : les 6 nombres bruts et les 3 index de catégorie ;
- `category_index` cherche la position de la valeur dans `cat_labels` (l'ordre du `StringIndexer`). Une valeur inconnue va dans la dernière case ;
- `date.isoweekday()` donne 1 pour lundi, comme `DAY_OF_WEEK` dans le dataset.

**e) Un garde-fou avant de prédire.** La taille de chaque vecteur encodé est comparée à `n_inputs` du modèle. Si `preprocessing.json` et les modèles viennent d'exports différents, la page affiche une erreur au lieu d'une probabilité fausse.

**f) Afficher le résultat**
1. Les modèles de `preprocessing.json` sont triés par AUC : le premier (XGBoost) donne la prédiction principale.
2. La page affiche sa probabilité, le verdict (« En retard » si la probabilité dépasse **son** seuil, 0,23) et une barre de progression.
3. Les autres modèles (MLP et RF) sont affichés en dessous, pour comparaison.

### Étape 7.5 : `explore.py`, explorer les retards
1. `require(DETAIL_CSV)`, puis lecture de `stats_detail.csv` : une ligne par (compagnie, mois, heure), avec `flights`, `delayed` et `sum_arr_delay`. Ces chiffres viennent des 5,2 M vols.
2. **Filtres** : compagnies (toutes si rien n'est choisi), plage de mois, plage d'heures, et choix de l'indicateur (taux de retard ou retard moyen).
3. **`aggregate(df, by)`** : on additionne d'abord les sommes, **puis** on divise (`delayed / flights`, `sum_arr_delay / flights`). Faire la moyenne de pourcentages déjà calculés serait faux, car un petit groupe pèserait autant qu'un gros.
4. **Affichage** : 3 chiffres clés (vols, taux de retard, retard moyen), un graphique par compagnie, un par heure, un par mois, et un tableau dépliable.

### Étape 7.6 : `model.py`, le modèle
1. `require(...)` sur `model_scores.csv` et `threshold_curves.csv`.
2. **Tableau des scores** : AUC, F1, accuracy et seuil de chaque modèle, triés par AUC. Une légende explique pourquoi l'accuracy est trompeuse (18 % de retards). Avec l'export du notebook bis, la régression logistique Spark y apparaît, avec son F1 de 0.
3. **Courbe du seuil** : pour le modèle choisi, la précision, le rappel et le F1 en fonction du seuil, tracés avec Altair. Une ligne pointillée marque le seuil retenu, et une phrase traduit : « le modèle détecte X % des retards, et Y % de ses alertes sont justes ».

La matrice de corrélation a été retirée de cette page. Elle reste dans l'exploration du notebook (partie 3.5), et la partie 8 explique comment la lire.

### Ce qu'il faut retenir sur l'app
1. **Séparation des rôles** : Spark fait le travail lourd une seule fois dans le notebook, l'app ne lit que de petits fichiers.
2. **Même encodage** : `encode()` relit `preprocessing.json`, et la vérification de taille bloque tout export incohérent.
3. **Des chiffres exacts** : les taux sont recalculés à partir de sommes, et le formulaire ne propose que des routes réelles.
4. **Limite** : l'app accepte des dates futures, mais le modèle n'a appris que sur 2015. Il utilise le mois, le jour et le jour de la semaine, pas l'année.
5. **Avec l'export du notebook bis**, la page Prédire montre 3 modèles (XGBoost, MLP et RF) : la régression logistique Spark n'est pas exportée, car l'app ne sait pas lui préparer une entrée Spark.

---

## Partie 8 : Lire la matrice de corrélation

**D'où elle vient** : `pdf[num_cols + ["label"]].corr()` dans l'export du notebook. C'est une corrélation de **Pearson**, calculée sur l'échantillon de 15 % (785 169 vols), entre les 6 variables numériques et la cible. Elle n'est plus affichée dans l'app, mais le notebook l'exporte toujours dans `correlation.csv`, et l'EDA calcule les mêmes liens (partie 3.5).

### Elle est juste
- Les valeurs concordent avec l'exploration (partie 3.5), qui était faite sur le dataset complet :
  - heure de départ / retard = **0,139**, contre 0,14 pour `SCHEDULED_DEPARTURE` dans l'EDA ;
  - mois / retard = -0,031, contre environ -0,04 dans l'EDA.
- La diagonale vaut 1, la matrice est symétrique, et les 7 variables sont renommées en français.

### Ce qu'elle montre
| Paire | Corrélation | Lecture |
|---|---|---|
| Heure de départ / retard | **0,139** | le seul lien notable : les retards s'accumulent au fil de la journée |
| Durée prévue / distance | **0,984** | presque la même information en double |
| Toutes les autres / retard | entre -0,03 et 0,005 | aucun lien **linéaire** |

### Ses limites (le prof peut les soulever)
1. **Pearson ne mesure que les liens linéaires.** Le mois et le jour de la semaine sont des codes **cycliques** : les retards montent en été **et** en décembre, donc une corrélation de -0,03 ne veut pas dire que le mois ne compte pas. Même chose pour l'heure, dont l'effet n'est pas une ligne droite. C'est pour ça que XGBoost et le MLP font mieux que la régression logistique.
2. **Les variables les plus utiles sont absentes** : la compagnie et les aéroports sont des catégories, et la corrélation de Pearson ne s'applique pas à elles. 
3. **Durée et distance (0,984)** : presque redondantes. Ce n'est pas une erreur. Les arbres s'en accommodent, et la différence entre les deux (le temps prévu au sol, au roulage) apporte un peu d'information.
4. **Cible 0/1** : la corrélation avec une variable binaire s'appelle *point-bisériale*. Elle est valable, mais reste faible par nature quand une classe ne représente que 18 %.

**Réponse type à l'oral** : « La matrice montre que les liens linéaires avec le retard sont faibles : seule l'heure de départ ressort, à 0,14. Ça ne veut pas dire que les autres variables sont inutiles : le mois et le jour ont des effets cycliques, et la compagnie et les aéroports sont des catégories absentes de la matrice. C'est pour ça que les modèles non linéaires comme XGBoost font mieux. »

---

## Partie 9 : Questions sur les combinaisons absentes du dataset

Le prof peut demander : « Et si la combinaison choisie n'existe pas dans le dataset ? » Les deux pages de l'app se comportent **très différemment**.

### Page « Explorer » : aucun modèle, seulement des comptages réels
- La page lit `stats_detail.csv` : les **vrais comptages** des vols 2015, une ligne par (compagnie, mois, heure) **qui existe**.
- Les filtres ne font que **sélectionner et additionner** ces lignes. Une combinaison qui n'existe pas n'a pas de ligne, donc elle ne compte pour rien.
- Si le filtre ne laisse aucune ligne, la page affiche **« Aucun vol pour ces filtres. »** (exemple vérifié : American Airlines (AA) à 3 h du matin).
- Seules **3 073** combinaisons existent sur les 4 032 possibles (14 compagnies × 12 mois × 24 heures).
- **Point faible** : certaines combinaisons ont très peu de vols (30 en ont moins de 10). Par exemple, JetBlue (B6) en février à minuit : **1 vol**, en retard, donc « 100 % de retard ». Le chiffre est exact, mais il ne veut rien dire. Le nombre de vols est visible dans le tableau dépliable.

### Page « Prédire » : c'est le modèle qui calcule
1. **La route et la compagnie existent toujours** : les menus enchaînés ne proposent que les **8 544** combinaisons départ / arrivée / compagnie de `routes.csv`. On ne peut pas demander un vol qui n'a jamais existé.
2. **La date et l'heure sont libres** : la combinaison exacte (cette route, cette compagnie, ce jour, cette heure) n'a très probablement jamais existé en 2015. Le modèle donne quand même une probabilité, car il ne **cherche pas** le vol dans les données : il **généralise** à partir de ce qu'il a appris sur chaque variable.

**Les limites** :
- **Routes rares** : 532 routes ont moins de 5 vols en 2015, et certaines un seul (ABQ → PHX avec AA). Le modèle s'appuie alors surtout sur les tendances générales (compagnie, heure, aéroports) plutôt que sur l'historique de la route.
- **Heure jamais desservie** : on peut demander AA à 3 h du matin. Le modèle répond à partir de ce qu'il a appris sur les vols de nuit en général : c'est une extrapolation.
- **Dates** : l'année est ignorée, seuls le mois, le jour et le jour de la semaine comptent. Le 29 février n'existe pas en 2015 : le modèle répond, mais n'a jamais vu ce jour.
- **Valeur inconnue** : si une compagnie ou un aéroport n'était pas dans les données d'entraînement, il irait dans la case « inconnu » du one-hot. Avec les menus, ça ne peut pas arriver.

### Réponses types
| Question du prof | Réponse |
|---|---|
| « Et si la combinaison n'existe pas dans le dataset ? » | « Sur Prédire, la route et la compagnie existent toujours : les menus sont construits à partir des routes réelles de 2015. La date et l'heure sont libres : le modèle ne cherche pas le vol, il généralise à partir des tendances apprises par variable. Sur Explorer, il n'y a pas de modèle : ce sont des comptages réels, et un filtre vide affiche "Aucun vol pour ces filtres". » |
| « Le modèle a-t-il déjà vu ce vol précis ? » | « Probablement pas, et ce n'est pas nécessaire : il a appris l'effet de l'heure, du mois, de la compagnie et des aéroports, et il les combine. C'est tout l'intérêt d'un modèle par rapport à un tableau de statistiques. » |
| « Quelle confiance pour une route rare ? » | « Moins de confiance : avec un seul vol en 2015, le modèle se base surtout sur la compagnie, l'heure et les aéroports, pas sur l'historique de la route. L'app affiche le nombre de vols de la compagnie sur la route sous le formulaire. » |
| « 100 % de retard sur Explorer, c'est fiable ? » | « Le chiffre est exact, mais il peut reposer sur un seul vol. Il faut regarder le nombre de vols dans le tableau avant de conclure. » |
| « Pourquoi accepter des dates futures ? » | « L'objectif est de prédire un vol à venir. Le modèle n'utilise que le mois, le jour et le jour de la semaine, appris sur 2015. Il suppose que les tendances de 2015 restent valables, ce qui est une limite. » |
| « Et un aéroport qui n'existait pas en 2015 ? » | « Il ne peut pas être choisi dans les menus. Dans le code, il irait dans la case "inconnu" prévue par `handleInvalid="keep"`. Le modèle répondrait alors sans information sur cet aéroport. » |

---

## Récapitulatif des résultats

| Modèle | Outil | Données de test | AUC | F1 retard | Seuil |
|---|---|---|---|---|---|
| **XGBoost** | xgboost | 157 033 vols | **0,724** | **0,43** | 0,23 |
| MLP | TensorFlow | 157 033 vols | 0,679 | 0,38 | 0,18 |
| Random Forest | Spark MLlib | 157 033 vols | 0,665 | 0,37 | 0,20 |
| Régression logistique | Spark ML Pipeline | 51 777 vols | 0,635 | 0 | 0,50 |
| Baseline (toujours « à l'heure ») | Spark | 51 777 vols | 0,500 | 0 | |

## Les phrases à placer à l'oral
1. « Spark traite les 5,2 millions de vols. Les modèles sont entraînés sur des échantillons de 5 et 15 %. »
2. « Deux modèles sont faits en `pyspark.ml` : la régression logistique, dans un Pipeline avec `randomSplit([0.8, 0.2], seed=42)`, et le Random Forest. »
3. « Aucune fuite de données : les colonnes connues après le départ sont retirées dès le nettoyage, la standardisation et les seuils sont calculés sans toucher au test. »
4. « On juge les modèles sur l'AUC, parce qu'avec 18 % de retards, l'accuracy récompense le modèle qui ne prédit jamais de retard. La baseline le montre : 81 % d'accuracy, mais une AUC de 0,5. »
5. « XGBoost est le meilleur, avec une AUC de 0,72. La régression logistique Spark classe correctement les vols (AUC 0,635), mais son seuil par défaut de 0,5 est trop haut pour des classes déséquilibrées. »

---

## Légende : vocabulaire et abréviations

### Big Data et Spark
| Terme | Signification |
|---|---|
| **HDFS** | système de fichiers distribué de Hadoop : les fichiers sont découpés en blocs de 128 Mo, répartis sur plusieurs machines |
| **NameNode** | le « chef » de HDFS : il sait quel fichier est découpé en quels blocs, et où ils sont. Il ne stocke aucune donnée |
| **DataNode** | une machine qui stocke les blocs. Ici il y en a 2, et chaque bloc est copié sur les deux (**réplication**) |
| **Driver** | le processus Spark qui pilote le calcul. En mode local, il fait aussi tout le travail |
| **DataFrame** | un tableau de données, avec des colonnes nommées et typées (Spark ou pandas) |
| **Transformation** | une opération Spark qui ne calcule rien tout de suite (`filter`, `select`, `withColumn`…) |
| **Action** | une opération qui déclenche le calcul (`count`, `show`, `first`, `toPandas`…) |
| **Évaluation paresseuse** (*lazy*) | Spark construit un plan et n'exécute rien avant une action |
| **Shuffle** | redistribution des données entre partitions (`dropDuplicates`, `groupBy`) : l'opération la plus coûteuse |
| **Cache / persist** | garder un DataFrame en mémoire pour ne pas tout recalculer à chaque action |
| **Partition** | un morceau du DataFrame, traité par un cœur du processeur |
| **Arrow** | un format de données par colonnes qui accélère le passage Spark → pandas |

### Préparation des données
| Terme | Signification |
|---|---|
| **EDA** | *Exploratory Data Analysis* : l'exploration des données avant de les modifier |
| **Cast** | conversion d'une colonne d'un type à un autre (texte → nombre) |
| **Outlier** | valeur extrême ou aberrante |
| **Feature** | une variable d'entrée du modèle (heure, distance, compagnie…) |
| **Label / cible** | ce qu'on veut prédire : 1 si le vol arrive avec plus de 15 min de retard, sinon 0 |
| **`StringIndexer`** | donne un numéro à chaque valeur texte (« AA » → 0) |
| **One-hot** | transforme un numéro en vecteur de 0 avec un seul 1, pour que le modèle ne croie pas que 2 > 1 a un sens |
| **`VectorAssembler`** | réunit plusieurs colonnes en une seule colonne vecteur, le format exigé par Spark ML |
| **Standardisation** | ramener une variable à une moyenne de 0 et un écart-type de 1 |
| **Pipeline** | une chaîne d'étapes (encodage puis modèle) apprises ensemble par un seul `fit()` |
| **Seed** (graine) | nombre qui fixe le hasard : même seed, même résultat |
| **Fuite de données** (*data leakage*) | utiliser une information qui ne serait pas disponible au moment de prédire (ex. `DEPARTURE_DELAY`) |

### Entraînement
| Terme | Signification |
|---|---|
| **Train / validation / test** | apprendre / régler (seuil, early stopping) / mesurer le score final, une seule fois |
| **`randomSplit`** | découpage aléatoire d'un DataFrame Spark (ici 80 % / 20 %) |
| **Époque** | un passage complet sur les données d'entraînement (réseaux de neurones) |
| **Early stopping** | arrêter l'entraînement quand le score de validation ne s'améliore plus |
| **Surapprentissage** (*overfitting*) | le modèle apprend le train par cœur et se trompe sur de nouvelles données |
| **Dropout** | éteindre des neurones au hasard pendant l'entraînement, contre le surapprentissage |
| **Régularisation** (`regParam`) | pénalité sur les poids trop grands, contre le surapprentissage |
| **Boosting** | des arbres construits à la suite, chacun corrigeant les erreurs des précédents (XGBoost) |
| **Bagging / forêt** | des arbres indépendants, entraînés sur des tirages différents, qui votent (Random Forest) |
| **Baseline** | modèle de référence très simple (ici : toujours « à l'heure ») qu'il faut battre |
| **MLP** | *Multi-Layer Perceptron* : réseau de neurones à couches cachées |

### Évaluation
| Terme | Signification |
|---|---|
| **Probabilité** | ce que sort le modèle : un nombre entre 0 et 1 |
| **Seuil** | au-dessus, on prédit « retard ». Il est choisi pour maximiser le F1 sur la validation |
| **TP** (vrai positif) | en retard, prédit en retard |
| **TN** (vrai négatif) | à l'heure, prédit à l'heure |
| **FP** (faux positif) | à l'heure, mais prédit en retard : une **fausse alerte** |
| **FN** (faux négatif) | en retard, mais prédit à l'heure : un **retard raté** |
| **Matrice de confusion** | le tableau `[[TN, FP], [FN, TP]]` |
| **Accuracy** | part des prédictions justes. Trompeuse ici : « toujours à l'heure » fait 81 % |
| **Précision** | parmi les vols annoncés en retard, la part vraiment en retard |
| **Rappel** (*recall*) | parmi les vols vraiment en retard, la part détectée |
| **F1** | moyenne harmonique de la précision et du rappel, sur la classe « retard » |
| **F1 pondéré** | moyenne des F1 des deux classes, pondérée par leur taille (évaluateur Spark). Tirée par la classe « à l'heure » |
| **AUC** | probabilité qu'un vol retardé ait un score plus haut qu'un vol à l'heure. 0,5 = hasard, 1 = parfait. Ne dépend pas du seuil |
| **Log-loss** | mesure d'erreur sur les probabilités, surveillée par l'early stopping de XGBoost |
| **Corrélation de Pearson** | lien **linéaire** entre deux variables, de -1 à 1. 0 = aucun lien linéaire |

### Abréviations et noms propres
| Abréviation | Signification |
|---|---|
| **BTS** | *Bureau of Transportation Statistics*, qui définit un vol en retard comme arrivant plus de 15 min après l'horaire |
| **USDOT** | ministère américain des transports, source du dataset Kaggle 2015 |
| **RF** | Random Forest (forêt aléatoire) |
| **TF** | TensorFlow |
| **ML** | *Machine Learning* (apprentissage automatique) |
| **HHMM** | format d'heure sur 4 chiffres (1745 = 17 h 45) |
