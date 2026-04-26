# Traitement statistique des données - Projet de classification de sentiments

Classification de sentiments de commentaires Twitter à l’aide de plusieurs classifieurs (Naive Bayes et SVM).

---

# Structure du projet

## 1. Corpus

### Corpus_original
- twitter_training.csv
- twitter_validation.csv  
Ces deux fichiers proviennent de datasets téléchargés en ligne.

### Corpus_traite
- training_deduplicate.csv : corpus d’entraînement utilisé pour SVM et Naive Bayes
- validation_clean.csv : corpus de validation utilisé pour l’évaluation
- training_deduplicate.arff : version ARFF pour WEKA (entraînement)
- validation_clean.arff : version ARFF pour WEKA (validation)

---

## 2. Prétraitement

Ce projet inclut trois étapes principales de prétraitement :

- suppression des emojis et nettoyage du texte
- suppression des doublons entre les corpus d’entraînement et de validation
- conversion des fichiers CSV au format ARFF pour WEKA

Toutes ces étapes sont regroupées dans les scripts suivants :

```
python3 pretraitement/csv_nettoie.py corpus_original/twitter_training.csv
python3 pretraitement/deduplicate.py corpus_original/twitter_training.csv
python3 pretraitement/convert_arff.py corpus/corpus_traite/training_deduplicate.csv
```

## 3. Résultats
- resultat_NB/ : résultats du modèle Naive Bayes (Python et WEKA)
- resultat_SVM/ : résultats du modèle SVM (Python et WEKA)

## 4. Modèles
- NB.py : script principal pour entraîner le modèle Naive Bayes
- SVM.py : script principal pour entraîner le modèle SVM

# 🚀 Utilisation

## Naive Bayes

Exemples de commandes :

```bash
python3 NB.py corpus/corpus_traite/training_deduplicate.csv corpus/corpus_traite/validation_clean.csv
python3 NB.py corpus/corpus_traite/training_deduplicate.csv corpus/corpus_traite/validation_clean.csv --alpha 0.5
python3 NB.py corpus/corpus_traite/training_deduplicate.csv corpus/corpus_traite/validation_clean.csv --alpha 2.0
python3 NB.py corpus/corpus_traite/training_deduplicate.csv corpus/corpus_traite/validation_clean.csv --class_prior "[0.25,0.25,0.25,0.25]"
python3 NB.py corpus/corpus_traite/training_deduplicate.csv corpus/corpus_traite/validation_clean.csv --alpha 0.5 --class_prior "[0.25,0.25,0.25,0.25]"
```

## SVM

```bash
python3 SVM.py  corpus/corpus_traite/training_deduplicate.csv corpus/corpus_traite/validation_clean.csv
python3 SVM.py  corpus/corpus_traite/training_deduplicate.csv corpus/corpus_traite/validation_clean.csv --n-split 3
python3 SVM.py  corpus/corpus_traite/training_deduplicate.csv corpus/corpus_traite/validation_clean.csv --C 0.2
python3 SVM.py  corpus/corpus_traite/training_deduplicate.csv corpus/corpus_traite/validation_clean.csv --C 2.0
python3 SVM.py  corpus/corpus_traite/training_deduplicate.csv corpus/corpus_traite/validation_clean.csv --class-weight balanced
```

