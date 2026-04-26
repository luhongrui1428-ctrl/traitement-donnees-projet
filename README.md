# Traitement statistique des données - Projet de classification de sentiments

Classification de sentiments de commentaires Twitter à l’aide de plusieurs classifieurs (Naive Bayes et SVM).

---

# 📁 Structure du projet

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

Ce dossier contient les scripts de prétraitement des données :

- csv_nettoie.py : suppression des emojis et nettoyage du texte
- deduplicate.py : suppression des doublons entre les corpus d’entraînement et de validation
- convert_arff.py : conversion des fichiers CSV au format ARFF pour WEKA

---

## 3. Résultats

- resultat_NB/ : résultats du modèle Naive Bayes (Python et WEKA)
- resultat_SVM/ : résultats du modèle SVM (Python et WEKA)

---

## 4. Modèles

- NB.py : script principal pour entraîner le modèle Naive Bayes
- SVM.py : script principal pour entraîner le modèle SVM

---

# 🚀 Utilisation

## Naive Bayes

Exemples de commandes :

```bash
python3 NB.py training_deduplicate.csv twitter_validation.csv
python3 NB.py training_deduplicate.csv twitter_validation.csv --alpha 0.5
python3 NB.py training_deduplicate.csv twitter_validation.csv --alpha 2.0
python3 NB.py training_deduplicate.csv twitter_validation.csv --class_prior "[0.25,0.25,0.25,0.25]"
python3 NB.py training_deduplicate.csv twitter_validation.csv --alpha 0.5 --class_prior "[0.25,0.25,0.25,0.25]"
