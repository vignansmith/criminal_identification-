from mtcnn.mtcnn import MTCNN
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
import os
os.environ['TF_ENABLE_ONEDNN_OPTS = 0']

mtcnn_model = MTCNN()
facenet_model = load_model('model/facenet_keras.h5')

#get face embedding using facenet
def get_embedding(face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    embedding = facenet_model.predict(samples)
    return embedding[0]

def extract_face(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    results = mtcnn_model.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    s1 = x1
    s2 = y1
    s3 = width
    s4 = height
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array, s1, s2, s3, s4


criminals = ['Emerald Elnas', 'Empoy Marquez', 'Johnny Alo', 'Jun Polo', 'Osama', 'Sean Batoon','Venkat']

labels = []
X = []
Y = []

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index        
    
path = "Dataset"

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name)
print(labels)

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if 'Thumbs.db' not in directory[j]:
            img = extract_face(root+"/"+directory[j])
            embedding = get_embedding(img)
            label = getID(name)
            X.append(embedding)
            Y.append(label)
            print(str(embedding.shape)+" "+name)

X = np.asarray(X)
Y = np.asarray(Y)

np.save('model/X.txt',X)
np.save('model/Y.txt',Y)

X = np.load('model/X.txt.npy')
Y = np.load('model/Y.txt.npy')

scaler = Normalizer(norm='l2')
X = scaler.fit_transform(X)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

print(X)
print(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

svm_cls = svm.SVC()
svm_cls.fit(X_train, y_train)
predict = svm_cls.predict(X_test)
acc = accuracy_score(y_test, predict)
print(acc)

print(y_test)
print(predict)



img, x1, y1, width, height = extract_face("testImages/1.jpg")
embedding = get_embedding(img)
test = []
test.append(embedding)
test = np.asarray(test)

test = scaler.transform(test)
predict = svm_cls.predict(test)
print(predict)
print(criminals[int(predict)])

img = cv2.imread("testImages/1.jpg")
cv2.rectangle(img, (x1, y1), (x1+width, y1+height), (0, 255, 0), 2)
cv2.imshow("aa", img)
cv2.waitKey(0)



            