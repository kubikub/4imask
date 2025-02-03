# Video Anonymizer

## Description

Video Anonymizer est une application qui permet d'anonymiser les visages dans les vidéos en utilisant différentes techniques telles que le floutage, la mosaïque et le masquage. L'application est construite avec PySide6 pour l'interface utilisateur et utilise OpenCV pour le traitement vidéo.

## Fonctionnalités

- Sélection de vidéo à anonymiser
- Choix du format de sortie (360p, 480p, 720p, 1080p)
- Options d'anonymisation : Mosaïque, Flou, Masque
- Pause, reprise et arrêt de l'anonymisation
- Visualisation en direct de l'anonymisation

## Prérequis

- Python 3.x
- PySide6
- OpenCV
- imutils
- skimage

## Installation

1. Clonez le dépôt :
    ```bash
    git clone https://github.com/4iTEC/4imask.git
    cd 4imask
    ```

2. Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

1. Lancez l'application :
    ```bash
    python anonimization.py
    ```

2. Utilisez l'interface pour :
    - Sélectionner une vidéo
    - Choisir le format de sortie
    - Sélectionner une méthode d'anonymisation
    - Démarrer, mettre en pause, reprendre ou arrêter l'anonymisation
    - Visualiser l'anonymisation en direct

## Contribuer

Les contributions sont les bienvenues ! Veuillez soumettre une pull request ou ouvrir une issue pour discuter des changements que vous souhaitez apporter.

## Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.