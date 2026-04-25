import pandas as pd
import argparse

# Configuration des colonnes
LABEL_COL = 2
TEXT_COL = 3

def main():
    parser = argparse.ArgumentParser(description="Purge des doublons et prévention de fuite de données")
    parser.add_argument("train_csv", help="Fichier d'entraînement original")
    parser.add_argument("test_csv", help="Fichier de test (référence pour la fuite)")
    args = parser.parse_args()

    # 1. Chargement des fichiers
    df_train = pd.read_csv(args.train_csv, header=None)
    df_test = pd.read_csv(args.test_csv, header=None)

    nb_initial = len(df_train)

    # 2. Suppression des doublons internes (Dédoublonnement)
    # On garde la première occurrence de chaque message unique
    df_train = df_train.drop_duplicates(subset=[TEXT_COL])
    nb_apres_interne = len(df_train)

    # 3. Suppression des messages présents dans le test (Anti-fuite)
    # On retire du train tout ce qui est considéré comme "examen" (test)
    test_texts_set = set(df_test[TEXT_COL].astype(str).tolist())
    df_train = df_train[~df_train[TEXT_COL].astype(str).isin(test_texts_set)]
    nb_final = len(df_train)

    # 4. Sauvegarde du nouveau fichier
    output_name = "training_clean.csv"
    df_train.to_csv(output_name, index=False, header=False)

    # 5. Affichage des statistiques
    print(f"--- Statistiques de nettoyage ---")
    print(f"Nombre initial : {nb_initial}")
    print(f"Doublons internes supprimés : {nb_initial - nb_apres_interne}")
    print(f"Messages de fuite (présents dans test) supprimés : {nb_apres_interne - nb_final}")
    print(f"Nombre final de lignes : {nb_final}")
    print(f"Fichier sauvegardé sous : {output_name}")

if __name__ == "__main__":
    main()
