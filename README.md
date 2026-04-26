# Traitement statistique des données - Projet de classification de sentiments

Classification de sentiments de commentaires Twitter à l’aide de plusieurs classifieurs : Naive Bayes et SVM.

---

# Structure du projet

L’arborescence du projet est la suivante :

```text
.
├── corpus
│   ├── corpus_original
│   │   ├── twitter_training.csv
│   │   └── twitter_validation.csv
│   └── corpus_traite
│       ├── training_clean.csv
│       ├── training_deduplicate.arff
│       ├── training_deduplicate.csv
│       ├── validation_clean.arff
│       └── validation_clean.csv
├── NB.py
├── pretraitement
│   ├── converter_arff.py
│   ├── csv_nettoie.py
│   └── deduplicate.py
├── README.md
├── resultat_NB
│   ├── Python
│   │   └── resultat_NB_Python.txt
│   └── WEKA
│       ├── NB_CrossValidation
│       ├── NB_SuppliedTestSet
│       └── NB_UseTrainingSet
├── resultat_SVM
│   ├── Python
│   │   └── resultat_SVM_Python.txt
│   └── WEKA
│       ├── SVM_CrossValidation.txt
│       ├── SVM_SuppliedTestSet.txt
│       └── SVM_UseTrainingSet.txt
└── SVM.py
```

---

## 1. Corpus

### `corpus_original`

- `twitter_training.csv`
- `twitter_validation.csv`

Ces deux fichiers proviennent de jeux de données téléchargés en ligne.

### `corpus_traite`

- `training_clean.csv` : corpus d’entraînement après nettoyage du texte
- `validation_clean.csv` : corpus de validation après nettoyage du texte
- `training_deduplicate.csv` : corpus d’entraînement utilisé pour SVM et Naive Bayes
- `training_deduplicate.arff` : version ARFF pour WEKA du corpus d’entraînement
- `validation_clean.arff` : version ARFF pour WEKA du corpus de validation

---

## 2. Prétraitement

Ce projet inclut trois étapes principales de prétraitement :

- suppression des emojis et nettoyage du texte ;
- suppression des doublons internes et prévention de la fuite de données entre le corpus d’entraînement et le corpus de validation ;
- conversion des fichiers CSV au format ARFF pour WEKA.

Toutes ces étapes sont regroupées dans les scripts suivants :

```bash
# Nettoyage du corpus d'entraînement -> sortie : training_clean.csv
python3 pretraitement/csv_nettoie.py corpus/corpus_original/twitter_training.csv corpus/corpus_traite/training_clean.csv

# Nettoyage du corpus de validation -> sortie : validation_clean.csv
python3 pretraitement/csv_nettoie.py corpus/corpus_original/twitter_validation.csv corpus/corpus_traite/validation_clean.csv

# Suppression des doublons internes et prévention de la fuite de données -> sortie : training_deduplicate.csv
python3 pretraitement/deduplicate.py corpus/corpus_traite/training_clean.csv corpus/corpus_traite/validation_clean.csv corpus/corpus_traite/training_deduplicate.csv

# Conversion du corpus d'entraînement au format ARFF pour WEKA -> sortie : training_deduplicate.arff
python3 pretraitement/converter_arff.py corpus/corpus_traite/training_deduplicate.csv corpus/corpus_traite/training_deduplicate.arff

# Conversion du corpus de validation au format ARFF pour WEKA -> sortie : validation_clean.arff
python3 pretraitement/converter_arff.py corpus/corpus_traite/validation_clean.csv corpus/corpus_traite/validation_clean.arff
```

---

## 3. Résultats

- `resultat_NB/` : résultats du modèle Naive Bayes avec Python et WEKA
- `resultat_SVM/` : résultats du modèle SVM avec Python et WEKA

---

## 4. Modèles

- `NB.py` : script principal pour entraîner et évaluer le modèle Naive Bayes
- `SVM.py` : script principal pour entraîner et évaluer le modèle SVM

---

# Utilisation

## Python

### Naive Bayes

Exemples de commandes :

```bash
python3 NB.py corpus/corpus_traite/training_deduplicate.csv corpus/corpus_traite/validation_clean.csv

python3 NB.py corpus/corpus_traite/training_deduplicate.csv corpus/corpus_traite/validation_clean.csv --alpha 0.5

python3 NB.py corpus/corpus_traite/training_deduplicate.csv corpus/corpus_traite/validation_clean.csv --alpha 2.0

python3 NB.py corpus/corpus_traite/training_deduplicate.csv corpus/corpus_traite/validation_clean.csv --class_prior "[0.25,0.25,0.25,0.25]"

python3 NB.py corpus/corpus_traite/training_deduplicate.csv corpus/corpus_traite/validation_clean.csv --alpha 0.5 --class_prior "[0.25,0.25,0.25,0.25]"
```

### SVM

Exemples de commandes :

```bash
python3 SVM.py corpus/corpus_traite/training_deduplicate.csv corpus/corpus_traite/validation_clean.csv

python3 SVM.py corpus/corpus_traite/training_deduplicate.csv corpus/corpus_traite/validation_clean.csv --n-splits 3

python3 SVM.py corpus/corpus_traite/training_deduplicate.csv corpus/corpus_traite/validation_clean.csv --C 0.2

python3 SVM.py corpus/corpus_traite/training_deduplicate.csv corpus/corpus_traite/validation_clean.csv --C 2.0

python3 SVM.py corpus/corpus_traite/training_deduplicate.csv corpus/corpus_traite/validation_clean.csv --class-weight balanced
```

### Résultats Python

Les résultats obtenus avec les scripts Python sont sauvegardés dans les dossiers suivants :

```text
resultat_NB/Python/resultat_NB_Python.txt
resultat_SVM/Python/resultat_SVM_Python.txt
```

## WEKA

Les fichiers utilisés avec WEKA sont les fichiers au format ARFF :

- `corpus/corpus_traite/training_deduplicate.arff` : corpus d'entraînement
- `corpus/corpus_traite/validation_clean.arff` : corpus de validation

Les expériences avec WEKA ont été réalisées avec le méta-classifieur `FilteredClassifier`.
Ce choix permet d'appliquer un filtre de transformation du texte avant l'entraînement du modèle.

### Utilisation avec l'interface graphique de WEKA

1. Ouvrir WEKA.
2. Aller dans `Explorer`.
3. Dans l'onglet `Preprocess`, ouvrir le fichier d'entraînement :

```text
corpus/corpus_traite/training_deduplicate.arff
```

4. Aller dans l'onglet `Classify`.
5. Choisir le classifieur :

```text
meta > FilteredClassifier
```

6. Dans `FilteredClassifier`, choisir le filtre :

```text
filters > unsupervised > attribute > StringToWordVector
```

7. Dans `FilteredClassifier`, choisir le classifieur de base :

```text
bayes > NaiveBayesMultinomial
```

ou, pour SVM :

```text
functions > SMO
```

8. Choisir le mode d'évaluation :

- `Use training set` pour tester sur le corpus d'entraînement ;
- `Cross-validation` pour faire une validation croisée ;
- `Supplied test set` pour évaluer avec le fichier de validation :

```text
corpus/corpus_traite/validation_clean.arff
```

9. Cliquer sur `Start` pour lancer l'expérience.

Les résultats obtenus avec WEKA sont sauvegardés dans les dossiers suivants :

```text
resultat_NB/WEKA/
resultat_SVM/WEKA/
```

### Résultats WEKA

Pour Naive Bayes, les résultats sont disponibles dans :

```text
resultat_NB/WEKA/NB_UseTrainingSet
resultat_NB/WEKA/NB_CrossValidation
resultat_NB/WEKA/NB_SuppliedTestSet
```

Pour SVM, les résultats sont disponibles dans :

```text
resultat_SVM/WEKA/SVM_UseTrainingSet.txt
resultat_SVM/WEKA/SVM_CrossValidation.txt
resultat_SVM/WEKA/SVM_SuppliedTestSet.txt
```
