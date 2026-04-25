import pandas as pd
import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# =========================
# 1. Command line arguments
# =========================
# On permet de régler alpha et class_prior depuis le terminal

parser = argparse.ArgumentParser(
    description="Classification de sentiments avec Naive Bayes + TF-IDF"
)

parser.add_argument("train_path", type=str, help="Chemin du fichier d'entraînement CSV")
parser.add_argument("test_path", type=str, help="Chemin du fichier de test CSV")

# Paramètre alpha (lissage)
parser.add_argument("--alpha", type=float, default=1.0,
                    help="Paramètre de lissage (alpha) pour Naive Bayes")

# Paramètre class_prior (optionnel)
parser.add_argument("--class_prior", type=str, default=None,
                    help="Prior des classes sous forme de liste, ex: [0.25,0.25,0.25,0.25]")

args = parser.parse_args()


# =========================
# 2. Load data
# =========================
train = pd.read_csv(args.train_path, header=None)
test = pd.read_csv(args.test_path, header=None)

# 0 = id, 1 = sujet, 2 = label, 3 = texte
X_train = train[3].astype(str)
y_train = train[2]

X_test = test[3].astype(str)
y_test = test[2]


# =========================
# 3. TF-IDF vectorization
# =========================
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    lowercase=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# =========================
# 4. Parse class_prior
# =========================

if args.class_prior is not None:
    class_prior = eval(args.class_prior)  # ex: "[0.25,0.25,0.25,0.25]"
else:
    class_prior = None


# =========================
# 5. Train Naive Bayes
# =========================
model = MultinomialNB(
    alpha=args.alpha,
    class_prior=class_prior
)

model.fit(X_train_vec, y_train)


# =========================
# 6. Prediction
# =========================
y_pred = model.predict(X_test_vec)


# =========================
# 7. Evaluation
# =========================
print("Accuracy:")
print(accuracy_score(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
