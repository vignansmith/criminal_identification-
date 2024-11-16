import tkinter as tk
from tkinter import messagebox, filedialog, Text, Scrollbar, Button, Label
import cv2
import numpy as np
import os
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
import seaborn as sns
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from skimage import feature
import pickle
from PIL import Image
import random
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Initialize global variables
mtcnn_model = MTCNN()
facenet_model = None
classifier = None
criminals = []
global X, Y, X_train, X_test, y_train, y_test, filename


# Dynamically generate the criminals list
def generate_criminals_list():
    global criminals, filename
    criminals = []
    if filename and os.path.exists(filename):
        criminals = [folder for folder in os.listdir(filename) if os.path.isdir(os.path.join(filename, folder))]
        text.insert(tk.END, "Criminals list updated: " + str(criminals) + "\n")
    else:
        text.insert(tk.END, "Error: Dataset folder not found or invalid.\n")


# Get criminal ID from name
def getID(name):
    return criminals.index(name) if name in criminals else -1


# Load dataset and FaceNet model
def uploadDataset():
    global filename, facenet_model
    filename = filedialog.askdirectory(initialdir="Dataset")
    text.delete('1.0', tk.END)

    if not filename:
        text.insert(tk.END, "No directory selected.\n")
        return

    generate_criminals_list()

    # Load FaceNet model
    facenet_model_path = 'model/facenet_keras.h5'
    if os.path.exists(facenet_model_path):
        try:
            facenet_model = load_model(facenet_model_path)
            text.insert(tk.END, "FaceNet Model Loaded Successfully.\n")
        except Exception as e:
            text.insert(tk.END, f"Error loading FaceNet model: {e}\n")
    else:
        text.insert(tk.END, "Error: FaceNet model not found at " + facenet_model_path + "\n")


# Preprocess dataset
def Preprocessing():
    global X, Y, X_train, X_test, y_train, y_test, filename
    text.delete('1.0', tk.END)

    if not filename or not os.path.exists(filename):
        text.insert(tk.END, "Error: Dataset not uploaded.\n")
        return

    X, Y = [], []
    for root, dirs, files in os.walk(filename):
        name = os.path.basename(root)
        label = getID(name)
        if label == -1:
            continue
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                img_resized = cv2.resize(img, (224, 224)) / 255.0  # Higher resolution (224x224)
                X.append(img_resized)
                Y.append(label)

    # Convert to numpy arrays
    X = np.array(X, dtype='float32')
    Y = np.array(Y)

    if len(X) == 0 or len(Y) == 0:
        text.insert(tk.END, "Error: No valid images found in the dataset.\n")
        return

    text.insert(tk.END, f"Preprocessing completed. Total images: {X.shape[0]}\n")



# Train SVM using HOG features
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def svmDeepFeatures():
    global X, Y, X_train, X_test, y_train, y_test, classifier

    if len(X) == 0 or len(Y) == 0:
        text.insert(tk.END, "Error: Preprocessing not completed.\n")
        return

    # Load MobileNetV2 model (pretrained on ImageNet) for feature extraction
    base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

    # Extract deep features
    deep_features = base_model.predict(X)

    # Normalize the features
    scaler = Normalizer().fit(deep_features)
    deep_features = scaler.transform(deep_features)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(deep_features, Y, test_size=0.2, random_state=42)

    # Train SVM classifier
    classifier = svm.SVC(probability=True, kernel='linear', C=1.0)
    classifier.fit(X_train, y_train)

    # Evaluate the model
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions) * 100
    precision = precision_score(y_test, predictions, average='macro') * 100
    recall = recall_score(y_test, predictions, average='macro') * 100
    fscore = f1_score(y_test, predictions, average='macro') * 100

    text.insert(tk.END, f"Deep Feature SVM Training Completed.\nAccuracy: {accuracy:.2f}%\nPrecision: {precision:.2f}%\nRecall: {recall:.2f}%\nF1-Score: {fscore:.2f}%\n")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", xticklabels=criminals, yticklabels=criminals, fmt="g")
    plt.title("Confusion Matrix for Deep Feature SVM")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()


# Predict a new image
def predict():
    global classifier, criminals

    if classifier is None:
        text.insert(tk.END, "Error: Classifier has not been trained.\n")
        return

    img_path = filedialog.askopenfilename(title="Select an image", filetypes=(("Image Files", "*.jpg;*.jpeg;*.png"),))
    if not img_path:
        text.insert(tk.END, "No image selected for prediction.\n")
        return

    # Load and preprocess the image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224)) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)

    # Extract deep features using the same model used for training
    base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    deep_features = base_model.predict(img_array)

    # Predict the criminal
    probabilities = classifier.predict_proba(deep_features)[0]
    prediction_index = np.argmax(probabilities)
    predicted_criminal = criminals[prediction_index]
    prediction_percentage = probabilities[prediction_index] * 100

    # Log results
    text.insert(tk.END, f"Predicted Criminal: {predicted_criminal}\n")
    text.insert(tk.END, f"Prediction Confidence: {prediction_percentage:.2f}%\n")

    # Draw bounding box and overlay text on the image
    display_img = cv2.resize(img, (800, 600))
    overlay_text = f"Predicted: {predicted_criminal} ({prediction_percentage:.2f}%)"
    cv2.rectangle(display_img, (50, 50), (750, 550), (0, 255, 0), 2)  # Green bounding box
    cv2.putText(display_img, overlay_text, (60, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)  # White overlay text

    # Show the prediction image
    cv2.imshow("Prediction", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# GUI Setup
main = tk.Tk()
main.title("Criminal Identification Using ML & Face Recognition")
main.geometry("1200x700")
main.configure(bg='darkviolet')

font = ('times', 15, 'bold')
title = Label(main, text="Criminal Identification Using ML & Face Recognition", font=font, bg='darkviolet', fg='gold')
title.pack()

font1 = ('times', 13, 'bold')
Button(main, text="Upload Dataset", command=uploadDataset, font=font1).place(x=50, y=100)
Button(main, text="Preprocessing", command=Preprocessing, font=font1).place(x=50, y=150)
Button(main, text="Train SVM (HOG)", command=svmDeepFeatures, font=font1).place(x=50, y=200)
Button(main, text="Predict", command=predict, font=font1).place(x=50, y=250)
Button(main, text="Exit", command=main.destroy, font=font1).place(x=50, y=300)

text = Text(main, height=25, width=80, font=('times', 12, 'bold'))
text.place(x=400, y=100)

main.mainloop()
