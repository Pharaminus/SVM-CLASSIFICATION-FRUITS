from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QLabel, QFrame, QWidget, QLineEdit, QPushButton, QMessageBox, QHBoxLayout, QVBoxLayout, QFileDialog, QProgressBar
from PyQt5.QtCore import QMetaObject, pyqtSlot
from PyQt5.QtGui import QPixmap
from Fruit_Classifier import Ui_FruitClasser
from PIL import Image
from skimage.color import rgb2gray, rgba2rgb
import numpy as np
from rembg import remove 
import io
from skimage.feature import hog
import joblib


class MainWindowBiblio(QMainWindow, Ui_FruitClasser):
    
    def __init__(self, parent=None):
        super(MainWindowBiblio,self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Classsifier")
        self.pushButtonImportClasser.clicked.connect(self.import_image)
        self.model1 = joblib.load("tp2_projet_classifier/svm_model_best.pkl")
        
        
        
        
        
    def import_image(self):
        # Chargement et stockage de l'image dans une variable
        self.labelTraitement.setText("Traitement...")
        image_path, _ = QFileDialog.getOpenFileName(self, "Sélectionner une image", "", "Images (*.png *.xpm *.jpg)")
        if image_path:
            
            self.image = Image.open(image_path)
            self.image_pixmap = QPixmap(image_path)
            
            self.zoneImage.setPixmap(self.image_pixmap.scaled(self.zoneImage.size()))
            proba = dict()
            reponse, proba = self.predict_image_class(image_path)
            # ["ananas", "avocat", "banane", "mangue", "pasteque"]
            # lb = QLabel()
                        
            print(int(proba['ananas'] * 100))
            self.progressBarAnanas.setValue(int(proba['ananas'] * 100))
            self.progressBarAvocat.setValue(int(proba['avocat'] * 100))
            self.progressBarBanane.setValue(int(proba['banane'] * 100))
            self.progressBarMangue.setValue(int(proba['mangue'] * 100))
            self.progressBarPasteque.setValue(int(proba['pasteque'] * 100))
            self.progressBarTraitementPrecision.setValue(int(proba[reponse[0]] * 100))
            self.labelResultat.setText(reponse[0])
            self.labelTraitement.setText("Precision de la prediction")
            
            
    def load_and_preprocess_image(self,image_path):
        img = Image.open(image_path)
        img= img.resize((128, 128))
        # buffer = io.BytesIO()
        # img.save(buffer, format="PNG")
        # img = buffer.getvalue()
        img = remove(img)
        
        # img = np.array(img)  # Conversion en tableau numpy
        return img
    
    # =============( extraction de caracteristique )==============
    def extract_hog_features_for_prediction(self,image):
        img1 = np.array(image)
        if image.format != "PNG" :
            if img1.shape[-1]  == 4:  # Si l'image a 4 canaux (RGBA), convertir en RGB
                image = rgba2rgb(image)
            else:
                image = rgba2rgb(image)
            
        gray_image = rgb2gray(image)
        features, _ = hog(gray_image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(2, 2), visualize=True)
        return features
    
    def predict_image_class(self, image_path):
        # =====================( Charger et prétraiter l'image )==================
        image = self.load_and_preprocess_image(image_path)
        # ===========( Extraire les caractéristiques HOG )======================
        features = self.extract_hog_features_for_prediction(image)
        
        #=============( Reshape pour respecter l'entrée du modèle )===============
        features = features.reshape(1, -1)
        
        # ==========( Prédire la classe )======================
        prediction = self.model1.predict(features)
        # l=[features.flatten()] 
        # probability=self.model1.predict_proba(l)
        
        proba = dict()
        val_proba = self.model1.predict_proba(features)
        val_proba = val_proba[0]
        proba["ananas"] = val_proba[0]
        proba["avocat"] = val_proba[1]
        proba["banane"] = val_proba[2]
        proba["mangue"] = val_proba[3]
        proba["pasteque"] = val_proba[4]
        
        # print(prediction)
        # ==========( Retourner le nom de la classe prédite )============
        return prediction, proba
    
    # =============( Exemple d'utilisation )=========================
    def predictElement(self):
        new_image_path = 'test/banane1.webp'
        predicted_class = self.predict_image_class(new_image_path)
        print(f'Predicted class:=======[ {predicted_class} ]==========')
        
        

