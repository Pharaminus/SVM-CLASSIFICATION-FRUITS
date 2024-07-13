
# Classification de Fruits avec SVM

Ce projet utilise un modèle de machine learning basé sur les Séparateurs à Vaste Marge (SVM) pour classifier différents types de fruits.

## Table des Matières
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Fonctionnalités](#fonctionnalités)
- [Méthodologie](#méthodologie)
- [Technologies Utilisées](#technologies-utilisées)
- [Contribution](#contribution)


## Installation


1. Installez les dépendances requises :
   ```
   sklearn
   rembg
   PIL
   joblib

   python 3.12.4
   ```

## Utilisation

1. Le code source en lui meme est document et fourni des reperes pour son utilisation `/`.
2. Exécutez le script principal :
   ```
   python classify_fruits.py
   ```
3. Le programme afficera les images pour chaque etape .

## Fonctionnalités

- Classification de 05 types de fruits différents (avocat, banane, pasteque, ananas et mangue)
- Modèle SVM entraîné sur un jeu de données d'images de fruits
- Visualisation des courbes d'apprentissage et de test
- Possibilité d'ajouter de nouvelles images de fruits pour la classification

## Méthodologie

1. Prétraitement des images (redimensionnement, normalisation des pixels)
2. Extraction des caractéristiques visuelles (couleurs, formes, textures)
3. Entraînement d'un modèle SVM sur un jeu de données d'images de fruits
4. Évaluation du modèle sur un ensemble de test
5. Optimisation des hyperparamètres du modèle SVM

## Technologies Utilisées

- Python
- Scikit-learn (pour le modèle SVM)
- NumPy (pour le traitement des données)
- Matplotlib (pour la visualisation)


