import argparse
import pandas as pd
import re

# Supprimer les emoji dans une cellule
def remove_emoji(text):
    if pd.isna(text):
        return text
    text = str(text)
    emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f"
    "]+", flags=re.UNICODE)

    return emoji_pattern.sub("", text)

# Nettoyage du corpus
def clean_csv(input_csv, output_csv):

    # Lire le CSV sans en-tête
    df = pd.read_csv(input_csv, header=None)

    nb_avant = len(df)

    # Supprimer les emoji dans la colonne des commentaires
    df[3] = df[3].map(remove_emoji)

    # Supprimer les doublons selon la colonne 0
    df = df.drop_duplicates(subset=[0], keep="first")

    nb_apres = len(df)

    # Sauvegarder le fichier nettoyé sans en-tête
    df.to_csv(output_csv, index=False, header=False)

    print(f"{output_csv} produit.")
    print(f"Lignes avant nettoyage : {nb_avant}")
    print(f"Lignes après nettoyage : {nb_apres}")
    print(f"Doublons supprimés : {nb_avant - nb_apres}")


def main():
    parser = argparse.ArgumentParser(
        description="Nettoyer le csv du corpus Twitter : suppression des emoji et des doublons par id."
        )

    parser.add_argument(
        "input_csv",
        help="Fichier CSV d'entrée."
        )

    parser.add_argument(
        "output_csv",
        help="Fichier CSV de sortie nettoyé."
        )

    args = parser.parse_args()

    clean_csv(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
