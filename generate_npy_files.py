import os
import cv2
import numpy as np

# List of criminal names corresponding to labels
criminals = ['ajmal kasab', 'Empoy Marquez', 'osama']

# Function to get the ID of a criminal based on the name
def getID(name):
    index = 0
    for i in range(len(criminals)):
        if criminals[i] == name:
            index = i
            break
    return index

# Function to process the dataset and save X and Y as .npy files
def preprocess_dataset(dataset_path):
    X = []
    Y = []
    
    # Walk through the dataset directory
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            name = os.path.basename(root)  # Folder name is the criminal name
            if 'Thumbs.db' not in file:    # Skip system files
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (32, 32))  # Resize image to 32x32
                    im2arr = np.array(img)           # Convert image to numpy array
                    im2arr = im2arr.reshape(32, 32, 3)  # Reshape to 32x32x3
                    label = getID(name)              # Get label (criminal ID)
                    X.append(im2arr)
                    Y.append(label)

    # Convert lists to numpy arrays
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Save the arrays as .npy files
    np.save('X.txt.npy', X)
    np.save('Y.txt.npy', Y)
    print(f"Preprocessing complete. Saved {X.shape[0]} images to 'X.txt.npy' and labels to 'Y.txt.npy'.")

# Main execution block
if __name__ == '__main__':
    dataset_path = input("Enter the path to the dataset folder: ")  # Input dataset directory
    if os.path.exists(dataset_path):
        preprocess_dataset(dataset_path)
    else:
        print(f"The folder '{dataset_path}' does not exist.")
