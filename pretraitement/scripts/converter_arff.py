#! /usr/bin/env python3

import argparse
import pathlib
import csv


# Paramètres fixes
TEXT_COL = 3 # commentaire
LABEL_COL = 2 # classe - sentiment
CLASS_LABELS = ["Negative", "Neutral", "Positive", "Irrelevant"]

# Nettoyer une chaîne de caractères pour le format ARFF
def nettoyage(s):
    s = str(s)
    s = s.replace("\\", "\\\\")
    s = s.replace("'", "\\'")
    s = s.replace("\r", " ")
    s = s.replace("\n", " ")
    return s


def csv_to_arff(input_file, output_file):
    rows = []
    bad_lines = []

    with open(input_file, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)

        for i, parts in enumerate(reader, start=1):

            sentiment = parts[LABEL_COL].strip()
            text = parts[TEXT_COL].strip()

            # Vérifier que la classe est bien connue
            if sentiment not in CLASS_LABELS:
                bad_lines.append((i, parts))
                continue

            rows.append((
                nettoyage(text),
                sentiment,
            ))

    relation_name = pathlib.Path(output_file).stem

    with open(output_file, "w", encoding="utf-8") as out:
        out.write(f"@relation {relation_name}\n\n")

        # Ordre fixe des attributs
        out.write("@attribute text string\n")
        out.write("@attribute xClasse {" + ",".join(CLASS_LABELS) + "}\n\n")

        out.write("@data\n")

        for text, sentiment in rows:
            out.write(f"'{text}',{sentiment}\n")

    print(f"ARFF écrit dans : {output_file}")
    print(f"Lignes valides : {len(rows)}")
    print(f"Lignes ignorées : {len(bad_lines)}")

def main():
    parser = argparse.ArgumentParser(
        description="Convertir un CSV Twitter nettoyé en ARFF avec attributs fixes."
        )

    parser.add_argument(
        "input_file",
        help="Fichier CSV d'entrée, sans en-tête."
        )

    parser.add_argument(
        "output_file",
        help="Fichier ARFF de sortie."
        )

    args = parser.parse_args()

    csv_to_arff(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
