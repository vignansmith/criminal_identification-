Criminal Identification Using ML & Face Recognition
This project uses machine learning and face recognition techniques to identify criminals from images. The system is based on extracting deep features using a pretrained FaceNet model and training a Support Vector Machine (SVM) classifier to recognize criminals. The system utilizes OpenCV, Keras, scikit-learn, and TensorFlow to build and deploy the model.

Features
Face Detection: Detects faces using the MTCNN face detector.
Face Recognition: Recognizes faces using the FaceNet deep learning model.
ML Classifier: Classifies criminals using SVM and KNN classifiers based on deep features extracted from the images.
Dataset Handling: Allows easy dataset upload and management.
Prediction: Allows users to predict a criminal from an image, displaying the predicted name and confidence level.
Graphical User Interface: Provides an intuitive GUI for easy interaction using Tkinter.
Requirements
To run this project, you need the following libraries installed:

Python 3.x
OpenCV (cv2)
Keras (tensorflow)
scikit-learn
PIL (Pillow)
MTCNN
NumPy
Matplotlib and Seaborn (for visualization)
TensorFlow
skimage
To install the required libraries, you can use pip:

bash
Copy code
pip install opencv-python
pip install keras
pip install scikit-learn
pip install pillow
pip install mtcnn
pip install numpy
pip install matplotlib
pip install seaborn
pip install tensorflow
pip install scikit-image
Project Setup
1. Clone the repository:
bash
Copy code
git clone https://github.com/your-username/criminal-identification-ml.git
cd criminal-identification-ml
2. Dataset Structure
The dataset should be organized as follows:

markdown
Copy code
Dataset/
    ├── Criminal_1/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
    ├── Criminal_2/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
    └── ...
Each folder under the Dataset/ directory should represent one criminal, and the images inside that folder should be images of that criminal.
You can add multiple criminals to the dataset, and the system will automatically recognize them.
3. Running the Application
After setting up the environment and the dataset, you can run the application by executing:

bash
Copy code
python main.py
This will launch the Tkinter-based GUI, where you can:

Upload Dataset: Select the folder containing the criminal images.
Preprocessing: Process the dataset to prepare the data for training.
Train SVM: Train an SVM classifier using deep features from the FaceNet model.
Predict: Select an image and the system will predict the criminal from the dataset based on the trained model.
Training and Prediction Workflow
1. Uploading Dataset:
Use the Upload Dataset button to select the folder containing images of criminals. The system will read the directory structure and update the list of criminals.
2. Preprocessing:
This step processes the images to extract features. The images are resized and normalized for training purposes.
3. Training the Model:
The Train SVM button will train an SVM classifier using the deep features extracted by the FaceNet model. The classifier will be trained using the features from the images and the corresponding labels (criminal names).
4. Prediction:
Use the Predict button to select an image. The system will predict the criminal identity based on the trained model and show the result with a confidence score.
Evaluation Metrics
The model's performance is evaluated using the following metrics:

Accuracy: The proportion of correctly classified images.
Precision: The proportion of true positives out of all predicted positives.
Recall: The proportion of true positives out of all actual positives.
F1-Score: The harmonic mean of precision and recall.
Confusion Matrix: Visualizes the performance of the classifier.
