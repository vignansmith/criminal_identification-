def Preprocessing():
    global X, Y, filename
    text.delete('1.0', END)
    
    if os.path.exists("model/X.txt.npy"):
        # Load preprocessed data if it exists
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        # Process images and create X, Y arrays
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:  # Skip system files
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (32, 32))  # Resize image
                    im2arr = np.array(img)           # Convert to numpy array
                    im2arr = im2arr.reshape(32, 32, 3)  # Reshape to 32x32x3
                    label = getID(name)              # Get label (criminal ID)
                    X.append(im2arr)
                    Y.append(label)
        # Convert to numpy arrays
        X = np.asarray(X)
        Y = np.asarray(Y)
        # Save the data
        np.save('model/X.txt', X)
        np.save('model/Y.txt', Y)
    text.insert(END,"Dataset Images Preprocessing completed\n")
    text.insert(END,"Total images found in Dataset : "+str(X.shape[0])+"\n")
