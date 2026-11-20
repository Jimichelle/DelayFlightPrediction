# Révision oral : Prédiction des retards de vols

> Méthode pour chaque réponse : **mot-clé + pourquoi + lien avec notre code**.
> Une réponse d'un seul mot ne suffit jamais.

## Les chiffres à connaître par cœur

| Quoi | Valeur |
|---|---|
| Lignes brutes (= Kaggle) | 5 819 079 × 31 colonnes |
| Lignes après nettoyage | 5 231 110 |
| Cible | `label = 1` si `ARRIVAL_DELAY > 15` min (définition du BTS) |
| Part des retards | ~18 % retardés / ~82 % à l'heure |
| Échantillon pour l'entraînement | 15 % (~780 000 vols) |
| Découpage | 70 % train / 10 % validation / 20 % test, seed 42 |
| Meilleure corrélation linéaire avec la cible | `SCHEDULED_DEPARTURE` ≈ 0,14 |

| Modèle (test set) | AUC | F1 | Seuil |
|---|---|---|---|
| **XGBoost** (meilleur) | **0,72** | 0,43 | 0,23 |
| MLP (TensorFlow) | 0,68 | 0,38 | 0,19 |
| Random Forest (Spark MLlib) | 0,66 | 0,37 | 0,20 |
| Régression logistique (TensorFlow) | 0,64 | 0,36 | 0,17 |

## Vocabulaire indispensable

- **NameNode** : stocke uniquement les **métadonnées** (quel fichier → quels blocs → sur quels DataNodes). Aucune donnée.
- **DataNode** : stocke les **blocs** de données (128 Mo par défaut).
- **Réplication** : chaque bloc est copié sur plusieurs DataNodes (`dfs.replication=2` chez nous).
- **Évaluation paresseuse (lazy)** : les transformations Spark construisent un plan (DAG). Seules les **actions** l'exécutent.
- **Shuffle** : redistribution des données entre partitions. C'est l'opération la plus coûteuse (`dropDuplicates`, `groupBy`).
- **Fuite de données (data leakage)** : utiliser une info qui ne serait pas disponible au moment de la prédiction.
- **Surapprentissage (overfitting)** : le modèle apprend le train par cœur et généralise mal.
- **AUC** : probabilité que le modèle donne un score plus élevé à un vol retardé qu'à un vol à l'heure. 0,5 = hasard, 1 = parfait. Ne dépend pas du seuil.
- **F1** : moyenne harmonique de la précision et du rappel, sur la classe « retard ».
- **Précision** : parmi les vols prédits en retard, combien le sont vraiment.
- **Rappel** : parmi les vols vraiment en retard, combien on en détecte.

---

## Partie 1 : Infrastructure HDFS

### Q1. Que se passe-t-il lors d'un `hdfs dfs -put` ? Pourquoi 2 DataNodes ?
1. Le client découpe le fichier en **blocs de 128 Mo**.
2. Pour chaque bloc, il demande au **NameNode** sur quels DataNodes l'écrire.
3. Le client envoie le bloc au **premier DataNode**, qui le **recopie sur le second** (pipeline de réplication).
4. Le NameNode ne garde que les métadonnées.

**2 DataNodes** : pour une vraie réplication (`dfs.replication=2`). Chaque bloc existe sur les 2 nœuds, donc si l'un tombe, le fichier reste lisible. Le cours n'en demandait qu'un.

### Q2. Comment Spark (sur la machine) accède-t-il aux DataNodes (dans Docker) ?
- Le NameNode renvoie le **nom d'hôte** des DataNodes (`datanode1`, `datanode2`).
- Il faut donc `127.0.0.1 namenode datanode1 datanode2` dans `/etc/hosts`,
- et `spark.hadoop.dfs.client.use.datanode.hostname=true` dans la session Spark.
- Chaque DataNode publie son propre port de transfert de blocs (9866 et 9966).

---

## Partie 2 : Lecture et parsing

### Q3. Pourquoi `inferSchema=false` ?
**Performance d'abord** : `inferSchema=true` oblige Spark à faire une **lecture complète supplémentaire** du fichier (5,8 M lignes, ~590 Mo) juste pour deviner les types.
On lit donc tout en texte, puis on **caste explicitement** en `double`. C'est plus rapide, et les types sont maîtrisés (pas de mauvais typage deviné).

### Q4. Pourquoi un échantillon de 5 % pour l'EDA ?
Les statistiques exploratoires n'ont pas besoin de 5,8 M lignes, 290 000 suffisent. Les comptages importants (valeurs distinctes, distribution de la cible) sont faits sur le dataset **complet**.

---

## Partie 3 : Nettoyage

### Q5. Pourquoi ne pas utiliser `DEPARTURE_DELAY` ? ⭐ (question quasi certaine)
C'est une **fuite de données**.
- On veut prédire **avant le départ**.
- `DEPARTURE_DELAY` n'est connu **qu'au décollage**. En production, on ne l'aurait pas.
- L'AUC serait excellente sur le papier, mais le modèle serait inutilisable en vrai.
- Même chose pour `TAXI_OUT`, `WHEELS_OFF`, `AIR_TIME`… Les colonnes `*_DELAY` (causes du retard) sont pire : elles sont **dérivées de la cible**.

On ne garde que : **compagnie, aéroports, mois, jour, jour de la semaine, heure prévue, durée prévue, distance**.

### Q6. Pourquoi retirer les vols annulés ou déroutés ?
Ils n'ont pas de `ARRIVAL_DELAY`, donc **pas de cible**. Les garder voudrait dire inventer une valeur.

### Q7. Pourquoi les colonnes `*_DELAY` ont-elles autant de valeurs manquantes ?
Elles ne sont remplies que quand le vol est **retardé** (pas annulé !). Les valeurs nulles sont donc normales.
⚠️ La cellule 17 du notebook dit « annulé » par erreur. À corriger.

### Q8. Pourquoi filtrer les codes d'aéroport numériques ?
En octobre 2015, une partie des aéroports sont notés avec un identifiant numérique à 5 chiffres au lieu du code à 3 lettres. On ne peut pas les relier à `airports.csv`, et ils dupliqueraient les catégories.

### Q9. Pourquoi garder `ARRIVAL_DELAY` entre -120 et 1500 min ?
Pour retirer les **valeurs extrêmes** (jusqu'à 1 971 min) qui sont des cas exceptionnels ou des erreurs. Les filtres sur les aéroports et ce plafond ont retiré 482 895 lignes.

### Q10. Pourquoi `.cache()` juste après `dropDuplicates()` ?
- Spark est **paresseux** : chaque **action** (`count`, `show`, `toPandas`, l'écriture des CSV) réexécute **toute la lignée** depuis la lecture HDFS.
- `dropDuplicates` déclenche un **shuffle**, l'opération la plus coûteuse.
- Toutes les étapes suivantes repartent de `df_clean`.
- Donc on le met en mémoire **une seule fois**, juste après. Sans ça, on referait lecture HDFS + shuffle à chaque action.
- Le `count()` après `cache()` sert aussi à **remplir** le cache (lui-même paresseux).
- On n'a qu'un seul cache, pour ne pas saturer la mémoire.

---

## Partie 4 : Features

### Q11. Comment est créée `dep_hour` ?
`SCHEDULED_DEPARTURE` est au format `HHMM`. On remplace `2400` par `0` (sinon on aurait une heure « 24 »), puis on divise par 100 → heure de 0 à 23.

### Q12. `StringIndexer` et `OneHotEncoder` : à quoi ça sert ?
- Les modèles ont besoin de **nombres**, pas de texte.
- `StringIndexer` : « AA » → 0, « DL » → 1… (les plus fréquents en premier).
- `OneHotEncoder` : index → vecteur de 0 et de 1, pour que le modèle ne croie pas que « DL (1) > AA (0) ».
- `handleInvalid="keep"` : une valeur **jamais vue** reçoit un index « inconnu » supplémentaire au lieu de faire planter le modèle (c'est le `+1` dans `n_cats`).

### Q13. Pourquoi standardiser seulement les colonnes numériques ?
Les réseaux de neurones apprennent mal quand les variables ont des échelles très différentes (la distance va jusqu'à plusieurs milliers, le jour de la semaine de 1 à 7). La standardisation donne moyenne 0 et écart-type 1. Les vecteurs one-hot sont déjà des 0/1. Les arbres (XGBoost, RF) n'en ont pas besoin.

---

## Partie 5 : Modèles et évaluation

### Q14. Pourquoi passer en pandas et ne pas tout faire en Spark ?
1. **Bibliothèques** : TensorFlow et XGBoost ne lisent pas les DataFrames Spark, il leur faut des tableaux NumPy sur une seule machine.
2. **Mémoire** : 5,2 M lignes × ~660 colonnes one-hot ne tiennent pas en RAM (le noyau plantait). D'où un **échantillon de 15 %**, et on ne ramène que 10 petites colonnes. Le one-hot est reconstruit en NumPy.
3. **Spark garde le travail lourd** : lecture, nettoyage et agrégations pour Streamlit sont faits sur **tout** le dataset.
4. Le **Random Forest** est entraîné avec **Spark MLlib**.
5. 780 000 lignes suffisent largement pour entraîner les modèles.

⚠️ Le cours demande « au moins 2 modèles `pyspark.ml` ». Nous en avons 1 (le RF), plus 3 autres modèles (TensorFlow est autorisé).

### Q15. Pourquoi 70 / 10 / 20 ?
- **Train (70 %)** : le modèle apprend.
- **Validation (10 %)** : on **règle** le modèle (choix du seuil, early stopping).
- **Test (20 %)** : score final, utilisé **une seule fois**, sur des données jamais vues.
- Si on réglait sur le test, le score serait **trop optimiste** (on tricherait).
- Les 4 modèles utilisent **exactement les mêmes lignes**, donc leurs scores sont comparables.

### Q16. Accuracy de 0,55 alors qu'un modèle « toujours non » fait 82 % : votre modèle est pire ? ⭐
**Non.**
- Classes **déséquilibrées** (18 % de retards) : le modèle « toujours non » a 82 % d'accuracy mais ne détecte **aucun** retard.
- On regarde l'**AUC** (le modèle constant : 0,5, nous : 0,64 à 0,72) et le **F1** sur la classe retard.
- L'accuracy est basse **par choix** : on a baissé le seuil (~0,2) pour détecter plus de retards, au prix de fausses alertes.

### Q17. Comment est choisi le seuil ? ⭐
- Le modèle sort une **probabilité** entre 0 et 1, et le seuil sert de coupure.
- À 0,5, le modèle ne prédit presque jamais de retard (classes déséquilibrées).
- `best_threshold` teste les seuils de 0,05 à 0,94 (pas de 0,01) et garde celui qui **maximise le F1 sur la validation**.
- **Jamais sur le test**, sinon on réglerait le modèle sur les données censées l'évaluer.

### Q18. À quoi sert `EarlyStopping` ?
Code : `EarlyStopping(monitor="val_loss", patience=2, min_delta=1e-4, restore_best_weights=True)`
- À chaque époque, on mesure la perte sur la **validation** (`val_loss`).
- Si elle ne s'améliore pas pendant **2 époques**, on arrête.
- On revient aux poids de la **meilleure** époque.
- But : éviter le **surapprentissage** et ne pas entraîner pour rien.
- XGBoost fait pareil : `early_stopping_rounds=20` sur la log-loss de validation.

### Q19. Pourquoi XGBoost est-il le meilleur ? 0,72, c'est bien ?
**Pourquoi XGBoost gagne** :
- La régression logistique est **linéaire** : elle additionne l'effet de chaque variable séparément.
- XGBoost : des **arbres** construits les uns après les autres, chacun corrige les erreurs des précédents (**boosting**). Les arbres captent les **interactions** (compagnie × aéroport × heure).

**0,72, c'est correct** :
- Les vraies causes des retards (météo, retard de l'avion précédent, contrôle aérien) **ne sont pas connues avant le départ**.
- L'EDA l'annonçait : la meilleure corrélation linéaire est de seulement 0,14.
- Une AUC proche de 1 avec nos variables serait **suspecte** et indiquerait une fuite de données.

### Q20. Régression logistique et MLP : quelle différence ?
- **Régression logistique** : une seule couche `Dense(1, sigmoid)`. C'est un modèle linéaire, qui sert de référence.
- **MLP** : couches cachées `Dense(128, relu)` → `Dense(64, relu)` avec `Dropout(0.2)`. Il apprend des relations **non linéaires**. Le Dropout éteint aléatoirement 20 % des neurones pendant l'entraînement, ce qui limite le surapprentissage.

### Q21. Random Forest : comment ça marche ? Pourquoi `maxDepth=10` ?
- 100 arbres, chacun entraîné sur une moitié aléatoire des vols (`subsamplingRate=0.5`) et un sous-ensemble aléatoire de variables à chaque coupure. La forêt fait voter les arbres.
- Il prend les entrées **brutes** (6 nombres + 3 index de catégorie), car les arbres n'ont besoin ni de standardisation ni de one-hot.
- `maxDepth=10` : à 12, la mémoire Java du driver ne suffisait plus (chaque niveau de plus double les statistiques à calculer). C'est aussi pour ça que le driver a 6 Go.

---

## Partie 6 : Application Streamlit

### Q22. L'app se connecte-t-elle à HDFS ?
**Non.** Elle lit uniquement les fichiers exportés par le notebook dans `app/data/` (CSV) et `app/models/` (modèles). Elle démarre un petit Spark local **seulement** pour charger le Random Forest.

### Q23. Comment garantir que l'app encode comme à l'entraînement ? ⭐
Avec **`preprocessing.json`**, exporté par le notebook :
- `mean` / `std` : moyenne et écart-type de chaque variable numérique, **calculés sur le train** ;
- `cat_labels` : l'ordre des catégories du **`StringIndexer`** (« AA » doit retomber sur le même index) ;
- un **slot « inconnu »** pour les valeurs jamais vues ;
- `models` : pour chaque modèle, son type d'entrée (`onehot` pour Keras/XGBoost, `raw` pour le RF) et son seuil.

`encode()` dans `predict.py` relit tout ça, et rien n'est écrit en dur dans l'app. Si on recalculait quoi que ce soit côté app, les prédictions seraient **fausses sans erreur visible**.

### Q24. Que montrent les 3 pages ?
- **Prédire** : origine, destination et compagnie (uniquement des routes réelles de 2015), date et heure → probabilité de retard selon les 4 modèles.
- **Explorer** : taux de retard et retard moyen par compagnie, heure et mois, avec des filtres.
- **Modèle** : scores sur le test, courbes précision / rappel / F1 en fonction du seuil, matrice de corrélation.

---

## Partie 7 : Questions supplémentaires

### Q25. Pourquoi la moyenne et l'écart-type sont-ils calculés sur le train seulement ?
Code : `mean = pdf.iloc[train_i][num_cols].mean()`.
- Si on les calculait sur tout l'échantillon, des infos du **test** entreraient dans la préparation des données : c'est une **petite fuite de données**.
- Le test doit simuler des données **jamais vues**. On applique donc au test (et à l'app) **les mêmes** `mean` et `std` que ceux du train.
- Même principe pour l'app : elle relit ces valeurs dans `preprocessing.json`.

### Q26. Lisez-moi une matrice de confusion. Quelle erreur est la plus grave ici ?
`evaluate_tf` affiche `[[TN, FP], [FN, TP]]` :
- **TN** : à l'heure, prédit à l'heure. **TP** : en retard, prédit en retard.
- **FP** (fausse alerte) : prédit en retard, mais à l'heure.
- **FN** (retard raté) : prédit à l'heure, mais en retard.
- Pour un passager, le **FN** est le plus gênant : on lui dit que tout va bien, et il est en retard. C'est pour ça qu'on baisse le seuil : on augmente le **rappel** (moins de FN), quitte à avoir plus de FP.

### Q27. Pourquoi `spark.sql.shuffle.partitions=16` ?
- Par défaut, Spark découpe chaque shuffle en **200 partitions**.
- En mode local sur 8 cœurs, 200 petites tâches coûtent surtout en **frais de gestion** (planification, petits fichiers intermédiaires).
- Avec 16 partitions (2 par cœur), chaque tâche a assez de travail et tous les cœurs sont occupés.

### Q28. Où est YARN dans votre projet ?
**Il n'y en a pas, et c'est un choix.**
- Spark tourne en **mode local** (`local[*]`) : le driver et l'exécution sont dans un seul processus Java, sur la machine, avec tous les cœurs.
- HDFS sert au **stockage**. YARN servirait à **répartir le calcul** sur plusieurs machines d'un cluster.
- Avec une seule machine, YARN n'apporterait rien. Sur un vrai cluster, on lancerait le même code avec `--master yarn`, sans rien changer d'autre.
- C'est aussi pour ça que la mémoire se règle avec `spark.driver.memory=6g` : en mode local, tout tourne dans le driver.

### Q29. Pourquoi Spark plutôt que MapReduce ?
- **MapReduce** écrit les résultats intermédiaires **sur disque** entre chaque étape (map → disque → reduce → disque).
- **Spark** garde les données **en mémoire** entre les étapes (`cache()`), et optimise tout le plan avant de l'exécuter (DAG).
- Pour notre notebook (nettoyage, puis plusieurs agrégations, puis le ML qui fait de nombreuses passes sur les mêmes données), MapReduce relirait le disque à chaque fois.
- Spark offre aussi une API haut niveau (DataFrames, SQL, MLlib) au lieu d'écrire des fonctions map et reduce à la main.

### Q30. Que se passe-t-il si un conteneur tombe ?
- **Un DataNode tombe** : aucune perte, chaque bloc a une copie sur l'autre (`dfs.replication=2`). Le NameNode le remarque (plus de *heartbeat*) et HDFS continue de servir les fichiers.
- **Le NameNode tombe** : HDFS est **inaccessible**, car plus personne ne sait où sont les blocs. C'est le **point de défaillance unique** (*single point of failure*) de notre installation.
- En production, on met la **haute disponibilité** : un NameNode actif et un en attente (*standby*). Pour un projet de développement, ce n'est pas nécessaire.
- Les données sont dans des **volumes Docker** : un redémarrage ne les efface pas. Seul `docker compose down -v` remet HDFS à zéro.

### Q31. Votre Random Forest reçoit les aéroports sous forme d'index. Il les traite comme des nombres ?
**Oui**, et il faut l'assumer honnêtement.
- Les index du `StringIndexer` (0, 1, 2…) passent par un `VectorAssembler` sur un DataFrame recréé depuis pandas. Spark ne sait plus que ce sont des catégories.
- Les arbres coupent donc sur des seuils du type « `origin_idx` < 37 ».
- Ça fonctionne quand même : les index sont **triés par fréquence** (0 = aéroport le plus fréquent), donc « petit index » veut dire « gros aéroport », ce qui a un sens. Et un arbre peut isoler une seule catégorie en faisant deux coupures.
- `maxBins=64` : nombre maximal d'intervalles testés pour chaque variable numérique à chaque coupure.
- Amélioration possible : déclarer ces colonnes comme catégorielles (métadonnées Spark), avec `maxBins` au moins égal au nombre d'aéroports (~320).

### Q32. Pourquoi `featureSubsetStrategy="sqrt"` et `subsamplingRate=0.5` ?
- Une forêt marche bien si les arbres sont **différents** les uns des autres : leurs erreurs se compensent quand on fait voter.
- `subsamplingRate=0.5` : chaque arbre voit une **moitié aléatoire** des vols (c'est le *bagging*). En plus, l'entraînement est 2 fois plus rapide.
- `featureSubsetStrategy="sqrt"` : à chaque coupure, l'arbre ne regarde qu'environ √9 = **3 variables** tirées au hasard. Sans ça, tous les arbres couperaient d'abord sur la même variable forte (l'heure) et se ressembleraient.

### Q33. Différence entre Random Forest et XGBoost ?
| | Random Forest | XGBoost |
|---|---|---|
| Construction | arbres **indépendants**, en parallèle | arbres **à la suite**, chacun corrige les erreurs des précédents |
| Combinaison | **vote** (moyenne) | **somme** des corrections |
| Ce qu'il réduit | la variance (instabilité) | le biais (erreurs systématiques) |
| Chez nous | AUC 0,66, Spark MLlib | AUC 0,72, `tree_method="hist"` |

`tree_method="hist"` : les valeurs de chaque variable sont regroupées en intervalles (histogrammes) avant de chercher les coupures. C'est beaucoup plus rapide sur des centaines de milliers de lignes. `QuantileDMatrix` prépare directement les données dans ce format.

### Q34. Comment l'app reste-t-elle rapide ? Et d'où viennent les routes proposées ?
- `@st.cache_data` pour les **données** (CSV) : lues une seule fois, puis gardées en mémoire entre les interactions.
- `@st.cache_resource` pour les **modèles** : chargés une seule fois (notamment la session Spark du RF, longue à démarrer).
- Sans cache, Streamlit relance tout le script à **chaque clic**, et rechargerait les 4 modèles à chaque fois.
- **Routes** : `routes.csv` contient uniquement les combinaisons origine, destination et compagnie **qui existent dans les données 2015**. L'utilisateur ne peut pas demander un vol qui n'existe pas. La durée et la distance prévues sont remplies automatiquement avec la **médiane** observée sur cette route.
- Les statistiques de la page Explorer sont calculées par Spark sur les **5,2 M vols**, pas sur l'échantillon.

---

## Partie 8 : Comment fonctionne le code de l'application Streamlit

### Le principe
L'app **ne calcule rien de lourd**. Elle lit les fichiers exportés par le notebook dans `app/data/` (CSV) et `app/models/` (modèles), puis les affiche ou s'en sert pour prédire. Elle ne se connecte ni à HDFS ni au dataset complet.

```
notebook --export--> app/data/*.csv  --> explore.py, model.py
                     app/models/*    --> predict.py
```

**À retenir sur Streamlit** : à **chaque interaction** (un clic, un menu, un curseur), Streamlit **relance tout le script de la page de haut en bas**. D'où les deux caches :

- `@st.cache_data` sur les fonctions qui lisent les CSV : le fichier est lu une seule fois, puis gardé en mémoire.
- `@st.cache_resource` sur `load_models` : les 4 modèles (et la session Spark du RF) sont chargés une seule fois. Sans ça, chaque clic rechargerait TensorFlow et redémarrerait Spark.

### `app.py` : le point d'entrée (15 lignes)
- `st.set_page_config` : titre de l'onglet, icône, mise en page large.
- `st.navigation([...])` : déclare les 3 pages et leur menu. `predict.py` est la page par défaut.
- `nav.run()` : exécute la page choisie.
- Avec `st.navigation`, Streamlit ne découvre **pas** tout seul le dossier `pages/` : pour ajouter une page, il faut l'ajouter dans cette liste.

### `predict.py` : prédire un retard

**a) Vérifications au démarrage.** Si `preprocessing.json`, `routes.csv` ou un modèle manque, la page affiche un message (« Relance l'export du notebook ») et s'arrête avec `st.stop()`, au lieu de planter.

**b) Chargement des modèles (`load_models`).** Il reconnaît le type de chaque modèle au nom de son fichier :

- un nom qui finit par `_spark` : Random Forest Spark. Il démarre une petite session Spark locale (`local[1]`, sans interface web), puis charge le modèle ;
- `.json` : XGBoost ;
- le reste (`.keras`) : TensorFlow.

**c) Le formulaire : uniquement des vols réels.** Les menus sont **chaînés** à partir de `routes.csv` :

1. **Départ** : tous les aéroports d'origine ;
2. **Arrivée** : seulement les destinations desservies depuis ce départ ;
3. **Compagnie** : seulement celles qui volent sur cette route, triées par nombre de vols.

On ne peut pas demander un vol qui n'existait pas en 2015. La **durée** et la **distance** ne sont pas demandées : elles sont lues dans `routes.csv` (médianes de cette route pour cette compagnie). L'utilisateur choisit seulement la **date** et l'**heure**.

**d) L'encodage (`encode()`), le cœur de la page.** Il refait **exactement** la préparation des données du notebook, à partir de `preprocessing.json` :

- **mode `onehot`** (Keras et XGBoost) : les 6 nombres sont standardisés avec la **moyenne et l'écart-type du train** (`(x - mean) / std`), puis chaque catégorie devient un vecteur one-hot avec une case de plus pour « inconnu » ;
- **mode `raw`** (Random Forest) : les 6 nombres bruts, suivis des 3 index de catégorie.

`category_index` donne la position de la compagnie ou de l'aéroport dans l'ordre du `StringIndexer`. Une valeur inconnue reçoit `len(labels)`, c'est-à-dire la case « inconnu ». Détail : `date.isoweekday()` donne 1 pour lundi, comme `DAY_OF_WEEK` dans le dataset.

**e) Un garde-fou.** Avant de prédire, la page compare la **taille du vecteur encodé** au nombre d'entrées attendu par chaque modèle (`n_inputs`). Si `preprocessing.json` et les modèles ne viennent pas du même export, elle affiche une erreur au lieu de donner une probabilité fausse.

**f) La prédiction.**

- `delay_proba` appelle la bonne méthode selon le modèle : `predictProbability` pour Spark, `predict` pour XGBoost et Keras.
- Dans `preprocessing.json`, les modèles sont **triés par AUC décroissante**. Le premier (XGBoost) donne la prédiction principale : la probabilité, puis le verdict « En retard » si elle dépasse **son** seuil (0,23).
- Les 3 autres modèles sont affichés en dessous, pour comparaison seulement.

### `explore.py` : explorer les retards
- Elle lit `stats_detail.csv` : une ligne par **(compagnie, mois, heure)** avec `flights`, `delayed` et `sum_arr_delay`. Ces chiffres ont été calculés par Spark sur les **5,2 M vols**.
- Les filtres (compagnies, plage de mois, plage d'heures) sélectionnent des lignes de ce fichier avec pandas.
- **`aggregate()`, point important** : on additionne d'abord les **sommes** (vols, vols retardés, minutes de retard), puis on divise. Faire la moyenne de taux déjà calculés serait faux, car un petit groupe pèserait autant qu'un gros. C'est pour ça que le notebook exporte des sommes et pas des pourcentages.
- La page affiche 3 chiffres clés (vols, taux de retard, retard moyen), puis des graphiques par compagnie, par heure et par mois, et un tableau dépliable.

### `model.py` : le modèle
- **Tableau des scores** (`model_scores.csv`) : AUC, F1, accuracy et seuil des 4 modèles, triés par AUC. Une légende explique pourquoi l'accuracy est trompeuse (18 % de retards).
- **Courbe du seuil** (`threshold_curves.csv`) : pour le modèle choisi, précision, rappel et F1 en fonction du seuil, tracés avec Altair. Une ligne pointillée marque le seuil retenu. La légende traduit en clair : « le modèle détecte X % des retards, et Y % de ses alertes sont justes ».
- **Matrice de corrélation** (`correlation.csv`) : carte de chaleur rouge et bleu entre les variables numériques et le retard. Elle montre que les liens linéaires sont faibles, ce qui explique pourquoi les modèles non linéaires gagnent.

### Les 3 idées à retenir pour l'oral
1. **Séparation des rôles** : Spark fait le travail lourd une seule fois dans le notebook, l'app ne fait que lire de petits fichiers.
2. **Cohérence de l'encodage** : `encode()` relit `preprocessing.json`, et la vérification de taille bloque tout export incohérent.
3. **Des chiffres exacts** : les stats sont recalculées à partir de sommes, et le formulaire ne propose que des routes réelles.

**Limite à connaître** : l'app accepte des dates futures, mais le modèle n'a appris que sur 2015. Il n'utilise que le mois, le jour et le jour de la semaine, pas l'année.

---

## Questions pièges

| Question | Réponse |
|---|---|
| « Le cours demande 2 modèles `pyspark.ml`, vous en avez combien ? » | Un, le Random Forest. TensorFlow et XGBoost font mieux et sont autorisés. Voir Q14. |
| « Le cours dit PySpark 3.5, vous utilisez 4.3 ? » | Spark 4 exige Java 17, donc on a pris Java 17/21 avec Spark 4 au lieu de Java 11 avec Spark 3.5. |
| « Le cours dit `randomSplit([0.8, 0.2])` ? » | On a besoin d'un jeu de validation en plus pour le seuil et l'early stopping, d'où 70/10/20. |
| « Votre AUC est faible. » | Voir Q19 : pas de fuite de données, causes réelles inconnues avant le départ. |
| « Quelqu'un d'autre obtiendrait-il les mêmes résultats ? » | Seeds fixées (42), README complet. Petites variations possibles avec TensorFlow (non déterministe sur certains calculs). |

## Erreurs que j'ai faites à la simulation

- ❌ Réponses d'un seul mot → toujours **mot-clé + pourquoi + lien avec le code**.
- ❌ « inferSchema = éviter les faux typages » → l'argument principal est la **lecture complète supplémentaire**.
- ❌ « Pandas parce que c'est trop long » → c'est la **mémoire** et le fait que **TF/XGBoost ne lisent pas Spark**.
- ❌ Seuil : je ne savais pas → **meilleur F1 sur la validation**.
- ❌ App : « via casting » → **`preprocessing.json`**.
- ❌ Dire « DataNode », pas « data note ».
