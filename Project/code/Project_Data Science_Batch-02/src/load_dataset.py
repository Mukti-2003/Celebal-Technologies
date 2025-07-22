import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
# ✅ Your dataset path
dataset_path = r"C:\Users\Admin\Downloads\archive (10)\animals\animals"

# Parameters
image_size = (128, 128)
features = []
labels = []

# Read all folders inside the dataset
for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)

    if not os.path.isdir(folder_path):
        continue  # Skip files, only process folders

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            image = cv2.resize(image, image_size)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # HOG feature extraction
            hog_feature = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), block_norm='L2-Hys')

            features.append(hog_feature)
            labels.append(folder_name)
        except Exception as e:
            print(f"Error with {image_path}: {e}")
            continue

# Convert to arrays and encode labels
X = np.array(features)
le = LabelEncoder()
y = le.fit_transform(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Dataset loaded successfully!")
print("Total samples:", len(X))
print("Classes:", le.classes_)
np.save('../outputs/features.npy', features)
np.save('../outputs/labels.npy', labels)