# APPLICATION DE CLASSIFICATION DE FRUIT UTILISANT l'approche de SVM
Projet de production et de deploiment d'un modele de classification de fruit(05 varietes : ananas
, banane, avocat, pasteque et mangue ). 
En utilisant plusieurs approche d'extraction de caracteristiques (HOG, SIFT et CNN)

## Getting Started
Instructions on how to get a copy of the project up and running on your local machine.
Pour avoir access au projet, il suffit de copier le lien du repository et de faire un gitclone
dans un repertoire de votre machine a partir du terminale. 

### Prerequisites
Librairies necessaire pour lancer le projet : 
 - **List 0** : Python 3.10.x

#### necessaire pour lancer le projet de deploiment(application)
 - **List 1** : PyQt5
 - **List 2** : rembg
 - **List 3** : numpy
 - **List 4** : joblib
#### necessaire pour executer les codes sources du projet
 - **List 1** : pandas
 - **List 2** : sklearn
 - **List 3** : matplotlib
 - **List 4** : tensorflow
 - **List 5** : matplotlib
 - **List 6** : glob
### Lancement du projet
L'ensemble du code source est ecrit dans des notes book il suffit juste d'executer chaque cellule.
### Lancement de l'application
Pour lancer l'application il faut se positionner dans le repertoire 
ou se trouve le fichier *start_app.py* et taper *python3 start_app.py* 


### Instalation
L'instalation des bibliotheques doivent se faire comme suit:
__pip install ``nom du module``__
 - ex : **pip install pandas**


## Acknowledgments
Proposer et coordonner par Dr Paulin Melatagia.

## Méthodologie

1. Prétraitement des images (redimensionnement, normalisation des pixels)
2. Extraction des caractéristiques visuelles (couleurs, formes, textures)
3. Entraînement d'un modèle SVM sur un jeu de données d'images de fruits
4. Évaluation du modèle sur un ensemble de test
5. Optimisation des hyperparamètres du modèle SVM