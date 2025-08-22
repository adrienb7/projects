# 📦 Installation de Docker sur Linux (via APT)

Ce guide détaille l'installation propre de Docker sur une distribution Linux basée sur Debian/Ubuntu, avec configuration personnalisée du répertoire Docker (`data-root`) et vérifications de bon fonctionnement.

---

## 🔁 1. Désinstallation d’anciennes versions (snap ou apt)

```bash
# Si Docker est installé via Snap
sudo snap remove docker

# Si Docker est installé via APT
sudo apt-get remove --purge docker docker-engine docker.io containerd runc
sudo apt-get autoremove
sudo apt-get autoclean

# Installer les dépendances
sudo apt-get update
sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Ajouter la clé GPG officielle Docker
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Ajouter le dépôt Docker
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Installation de Docker
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Éditez le fichier daemon.json
sudo nano /etc/docker/daemon.json


{
  "data-root": "/mnt/data/docker"
}

sudo mkdir -p /mnt/data/docker
sudo chown root:root /mnt/data/docker


sudo systemctl daemon-reexec
sudo systemctl restart docker
sudo systemctl enable docker


sudo usermod -aG docker $USER

docker ps


docker info | grep "Docker Root Dir"


docker system prune -a

