import csv
import json
import argparse

# Champs dans l'ordre du modèle
FIELDS = [
    "jour", "mois", "an", "hrmn", "lum", "agg", "atm", "col",
    "lat", "long", "catr", "circ", "nbv", "vosp", "prof", "plan", "surf",
    "infra", "situ", "vma", "senc", "catv", "obs", "obsm", "choc", "manv",
    "motor", "place", "catu", "sexe", "an_nais", "secu1", "secu2", "secu3", "locp"
]

def build_payload(row):
    values = [float(val) for val in row]
    if len(values) != len(FIELDS):
        raise ValueError(f"Ligne invalide : {len(values)} valeurs, attendu : {len(FIELDS)}")
    return json.dumps({field: value for field, value in zip(FIELDS, values)}, indent=2)

def generate_curl(payload, model_name, api_key, url):
    return f"""curl -X POST "{url}/predict/{model_name}" \\
  -H "Content-Type: application/json" \\
  -H "api_key: {api_key}" \\
  -d '{payload}'"""

def main(csv_path, model_name, api_key, url, output):
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        lines = list(reader)

    curls = []
    for i, row in enumerate(lines, 1):
        try:
            payload = build_payload(row)
            curl_cmd = generate_curl(payload, model_name, api_key, url)
            curls.append(curl_cmd)
        except Exception as e:
            print(f"⚠️ Ligne {i} ignorée : {e}")

    if output:
        with open(output, "w") as f:
            for cmd in curls:
                f.write(cmd + "\n\n")
        print(f"✅ {len(curls)} commandes curl écrites dans '{output}'")
    else:
        for cmd in curls:
            print(cmd)
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère des commandes curl pour FastAPI depuis un CSV.")
    parser.add_argument("csv_path", help="Chemin vers le fichier CSV")
    parser.add_argument("--model", default="gravite_accidents", help="Nom du modèle")
    parser.add_argument("--api_key", default="YOUR_API_KEY", help="Clé API")
    parser.add_argument("--url", default="http://localhost:8000", help="URL de l'API")
    parser.add_argument("--output", help="Fichier .sh pour sauvegarder les commandes")

    args = parser.parse_args()
    main(args.csv_path, args.model, args.api_key, args.url, args.output)
