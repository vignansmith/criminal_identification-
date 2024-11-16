import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Define the directory containing the folders for each criminal
dataset_dir = 'C:/Users/SMITH/OneDrive/Desktop/Coding/Vignan Mini Project/Dataset'  # Replace with the path to your dataset folder

# Initialize empty lists for storing image data and labels
X = []
Y = []

# List of criminals is inferred from folder names
criminals = os.listdir(dataset_dir)

# Function to map folder names to labels (criminals)
def getID(name):
    try:
        return criminals.index(name)  # Return the index of the criminal in the list
    except ValueError:
        return -1  # If the folder doesn't exist in the list, return -1

# Loop through each criminal folder and process the images
for criminal in criminals:
    criminal_folder = os.path.join(dataset_dir, criminal)
    
    if os.path.isdir(criminal_folder):  # Check if it is a directory (folder)
        print(f"Processing folder: {criminal_folder}")
        for file in os.listdir(criminal_folder):
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):  # Add more extensions if needed
                image_path = os.path.join(criminal_folder, file)
                print(f"Found image: {image_path}")  # Debugging line
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Warning: Could not read image {file} in {criminal_folder}. Skipping this image.")
                    continue
                img = cv2.resize(img, (32, 32))  # Resize to a consistent size (you can change this)
                img_array = np.array(img)
                X.append(img_array)
                Y.append(getID(criminal))

# Convert lists to numpy arrays
X = np.array(X)
Y = np.array(Y)

# Check if data has been loaded
if X.shape[0] == 0 or Y.shape[0] == 0:
    print("Error: No images were loaded. Please check your dataset.")
else:
    print(f"Successfully loaded {X.shape[0]} images.")

# Save X.txt.npy and Y.txt.npy
np.save('model/X.txt.npy', X)
np.save('model/Y.txt.npy', Y)

print("X.txt.npy and Y.txt.npy have been saved successfully.")

# Split the data into training and testing (80% training, 20% testing)
if X.shape[0] > 0:  # Only split if there are samples
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # Save the training data
    np.save('model/X_train.npy', X_train)
    np.save('model/Y_train.npy', Y_train)

    # Save the testing data (optional, in case you need it later)
    np.save('model/X_test.npy', X_test)
    np.save('model/Y_test.npy', Y_test)

    print("X_train.npy and Y_train.npy have been saved successfully.")
else:
    print("Error: No images were available for training. Please check your dataset.")
