import argparse
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Colonnes fixes du CSV
# 0 = id
# 1 = entity
# 2 = sentiment - classe
# 3 = commentaire
LABEL_COL = 2
TEXT_COL = 3

# Charger un fichier CSV nettoyé
# On n'utilise que la colonne 3 comme texte et la colonne 2 comme classe
def charger_corpus(fichier_csv):
    df = pd.read_csv(fichier_csv, header=None)

    # Supprimer les lignes où le texte ou la classe est vide
    df = df.dropna(subset=[LABEL_COL, TEXT_COL])

    textes = df[TEXT_COL].astype(str)
    labels = df[LABEL_COL].astype(str)

    return textes, labels

def main():

    parser = argparse.ArgumentParser(description="Classification SVM linéaire et SVM à noyau RBF")

    parser.add_argument(
        "train_csv", help="Fichier CSV d'entraînement nettoyé")

    parser.add_argument(
        "test_csv", help="Fichier CSV de test nettoyé")

    # nombre de folds pour la validation croisée
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Nombre de folds pour la validation croisée sur le train")

    # nombre maximal d'itérations pour LinearSVC
    parser.add_argument("--max-iter", type=int, default=50000,
                        help="Nombre maximal d'itérations pour LinearSVC")

    # paramètre de régularisation C
    parser.add_argument("--C", type=float, default=1.0,
                        help="Paramètre de régularisation (C)")

    # gestion du déséquilibre des classes
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                        help="Gestion du déséquilibre des classes")

    args = parser.parse_args()

    # 1. Charger le train et le test séparément
    textes_train, y_train = charger_corpus(args.train_csv)
    textes_test, y_test = charger_corpus(args.test_csv)

    # 2. Vectorisation TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        lowercase=True)

    X_train = vectorizer.fit_transform(textes_train)
    X_test = vectorizer.transform(textes_test)

    # 3. Validation croisée stratifiée sur le train uniquement
    cv = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=42
    )

    # définition du poids des classes
    class_weight_value = "balanced" if args.class_weight == "balanced" else None

    # 4. Modèle SVM linéaire
    clf_linear = LinearSVC(
        C=args.C,
        max_iter=args.max_iter,
        random_state=42,
        class_weight=class_weight_value
    )

    # 5. Modèle SVM à noyau RBF (non linéaire)
    clf_rbf = SVC(
        kernel="rbf",  # noyau gaussien
        C=args.C,
        gamma="scale",
        random_state=42,
        class_weight=class_weight_value
    )

    print("Paramètres utilisés :")
    print("n_splits =", args.n_splits)
    print("max_iter =", args.max_iter)
    print("C =", args.C)
    print("class_weight =", class_weight_value)
    print("vectorisation = TF-IDF")
    print("max_features = 10000")
    print("ngram_range = (1, 3)")

    # 6. Validation croisée - modèle linéaire
    cv_scores_linear = cross_validate(
        clf_linear,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
            "f1_macro": "f1_macro"
        }
    )

    print("\n================ MODELE LINEAIRE ================")
    print("Accuracy moyenne :", cv_scores_linear["test_accuracy"].mean())
    print("F1 macro moyen :", cv_scores_linear["test_f1_macro"].mean())

    # 7. Validation croisée - modèle RBF
    cv_scores_rbf = cross_validate(
        clf_rbf,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
            "f1_macro": "f1_macro"
        }
    )

    print("\n================ MODELE RBF ================")
    print("Accuracy moyenne :", cv_scores_rbf["test_accuracy"].mean())
    print("F1 macro moyen :", cv_scores_rbf["test_f1_macro"].mean())

    # 8. Entraînement final + test - modèle linéaire
    clf_linear.fit(X_train, y_train)
    y_pred_linear = clf_linear.predict(X_test)

    print("\n================ TEST LINEAIRE ================")
    print("Accuracy :", accuracy_score(y_test, y_pred_linear))
    print(classification_report(y_test, y_pred_linear))
    print(confusion_matrix(y_test, y_pred_linear))

    # 9. Entraînement final + test - modèle RBF
    clf_rbf.fit(X_train, y_train)
    y_pred_rbf = clf_rbf.predict(X_test)

    print("\n================ TEST RBF ================")
    print("Accuracy :", accuracy_score(y_test, y_pred_rbf))
    print(classification_report(y_test, y_pred_rbf))
    print(confusion_matrix(y_test, y_pred_rbf))


if __name__ == "__main__":
    main()
