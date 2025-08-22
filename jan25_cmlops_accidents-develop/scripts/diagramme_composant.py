import graphviz
import os

# Chemin de sortie (modifie ici selon tes besoins)
output_dir = '.'
output_filename = 'docker_compose_diagram'

# Créer le dossier s’il n'existe pas
os.makedirs(output_dir, exist_ok=True)

# Initialiser le diagramme
dot = graphviz.Digraph(comment='Docker Compose Diagram', format='png')
dot.attr(rankdir='LR', fontsize='10')

# Réseaux
dot.node('front-tier', 'Network: front-tier', shape='ellipse', style='filled', fillcolor='lightblue')
dot.node('back-tier', 'Network: back-tier', shape='ellipse', style='filled', fillcolor='lightblue')

# Services
services = {
    'postgresql_accidents': ['back-tier'],
    'adminer': ['back-tier', 'front-tier'],
    'app': ['back-tier', 'front-tier'],
    'mlflow-server': ['back-tier', 'front-tier'],
    'minio': ['back-tier', 'front-tier'],
    'redis': ['back-tier'],
    'airflow-webserver': ['back-tier', 'front-tier'],
    'airflow-scheduler': ['back-tier', 'front-tier'],
    'airflow-worker': ['back-tier', 'front-tier'],
    'airflow-init': ['back-tier', 'front-tier'],
    'flower': ['back-tier', 'front-tier'],
}

for svc in services:
    dot.node(svc, svc, shape='box', style='filled', fillcolor='lightgray')

# Connexions réseau
for svc, nets in services.items():
    for net in nets:
        dot.edge(svc, net)

# Dépendances
dot.edge('app', 'mlflow-server', label='depends_on')
dot.edge('app', 'postgresql_accidents', label='depends_on')
dot.edge('mlflow-server', 'minio', label='depends_on')
dot.edge('mlflow-server', 'postgresql_accidents', label='depends_on')
dot.edge('adminer', 'postgresql_accidents', label='depends_on')
dot.edge('airflow-webserver', 'postgresql_accidents', label='depends_on')
dot.edge('airflow-webserver', 'redis', label='depends_on')

# 👉 Chemin complet pour le fichier de sortie (sans extension)
output_path = os.path.join(output_dir, output_filename)

# Générer le PNG
dot.render(output_path, format='png', cleanup=True)

print(f"Diagramme généré ici : {output_path}.png")
