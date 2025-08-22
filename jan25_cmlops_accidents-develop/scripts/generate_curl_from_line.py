import sys

# 1. Liste des champs dans l’ordre du modèle PredictAccidentsPayload
FIELDS = [
    "jour", "mois", "an", "hrmn", "lum", "agg", "atm", "col",
    "lat", "long", "catr", "circ", "nbv", "vosp", "prof", "plan", "surf",
    "infra", "situ", "vma", "senc", "catv", "obs", "obsm", "choc", "manv",
    "motor", "place", "catu", "sexe", "an_nais", "secu1", "secu2", "secu3", "locp"
]

def build_payload_from_line(line: str) -> str:
    values = [float(val) for val in line.strip().split()]
    if len(values) != len(FIELDS):
        raise ValueError(f"Nombre de valeurs incorrect : {len(values)} trouvé, {len(FIELDS)} attendu.")
    
    payload = {field: value for field, value in zip(FIELDS, values)}
    
    import json
    return json.dumps(payload, indent=2)

def generate_curl(payload: str, model_name="gravite_accidents", api_key="YOUR_API_KEY", url="http://localhost:8000"):
    return f"""curl -X POST "{url}/predict/{model_name}" \\
  -H "Content-Type: application/json" \\
  -H "api_key: {api_key}" \\
  -d '{payload}'"""

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_curl.py \"<ligne de données tabulées séparées par des espaces/tabulations>\"")
        sys.exit(1)

    line = sys.argv[1]
    payload_json = build_payload_from_line(line)
    curl_cmd = generate_curl(payload_json)
    print(curl_cmd)
