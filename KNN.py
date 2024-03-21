import os
import random
import cv2
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split

# Constants
IMAGE_SIZE = (128, 128)
OVERSAMPLING_STRATEGY = "minority"
RANDOM_STATE = 42

def get_pixel(img, center, x, y):
    try:
        return 1 if img[x][y] >= center else 0
    except IndexError:
        return 0

def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = [
        get_pixel(img, center, x-1, y-1),
        get_pixel(img, center, x-1, y),
        get_pixel(img, center, x-1, y + 1),
        get_pixel(img, center, x, y + 1),
        get_pixel(img, center, x + 1, y + 1),
        get_pixel(img, center, x + 1, y),
        get_pixel(img, center, x + 1, y-1),
        get_pixel(img, center, x, y-1)
    ]
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    return sum(val_ar[i] * power_val[i] for i in range(len(val_ar)))

def calculate_lbp_features(image):
    rows, cols = image.shape
    lbp_values = np.zeros_like(image, dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            lbp_values[i, j] = lbp_calculated_pixel(image, i, j)

    return lbp_values.flatten()

def extract_lbp_features(file_path):
    try:
        image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.resize(gray_image, IMAGE_SIZE)
        lbp_features = calculate_lbp_features(gray_image)
        return lbp_features

    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
        return None

def load_dataset(folder_path, label):
    features = []
    labels = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, filename)
            lbp_features = extract_lbp_features(file_path)

            if lbp_features is not None:
                features.append(lbp_features)
                labels.append(label)

    return features, labels

def load_all_features_and_labels(folder_paths, samples_per_class=500):
    all_features = []
    all_labels = []
    total_images = 0

    for folder_path in folder_paths:
        print(f"Processing folder: {folder_path}")
        num_images_in_folder = sum(1 for _ in os.listdir(folder_path) if _.lower().endswith(('.png', '.jpg', '.jpeg')))
        total_images += num_images_in_folder
        label = 1 if "healthy" in folder_path.lower() else 0
        features, labels = load_dataset(folder_path, label)

        # Perform undersampling to limit the number of samples per class
        if len(features) > samples_per_class:
            selected_indices = random.sample(range(len(features)), samples_per_class)
            features = [features[i] for i in selected_indices]
            labels = [labels[i] for i in selected_indices]

        all_features.extend(features)
        all_labels.extend(labels)

    print(f"Total number of images: {total_images}")
    return all_features, all_labels


def train_knn_model(features, labels, n_neighbors=3, weights='distance', metric='euclidean', test_size=0.2, cv=5, random_state=RANDOM_STATE):
    print(f"n_neighbors={n_neighbors}, weights={weights}, metric={metric}")


    # Oversampled the features by multiplying the original value with a number between 0 and 1
    smote = SMOTE(sampling_strategy=OVERSAMPLING_STRATEGY, random_state=random_state)
    features, labels = smote.fit_resample(features, labels)


    # Standardize the features
    scaler = StandardScaler()
    features = scaler.fit_transform(np.array(features))

    # Initialize KNN model
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)

    # Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    # Split the data into training and testing sets before oversampling
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size,
                                                        random_state=random_state)

    # Cross-validated scores
    cross_val_scores = cross_val_score(knn_model, X_train, y_train, cv=skf)

    print(f"Cross-Validation Scores: {cross_val_scores}")
    print(f"Mean Cross-Validation Score: {np.mean(cross_val_scores):.2f}")

    # Train the KNN model on the entire resampled and standardized training set
    knn_model.fit(X_train, y_train)

    display_model_performance(knn_model, X_test, y_test)

    return knn_model, scaler


def display_model_performance(knn_model, X_test_normalized, y_test):
    y_prediction = knn_model.predict(X_test_normalized)
    print("Classification Report:")
    print(classification_report(y_test, y_prediction))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_prediction))
    accuracy = knn_model.score(X_test_normalized, y_test)
    print(f"\nAccuracy: {accuracy:.2f}")



def save_model_and_scaler(knn_model, scaler, model_filename='knn_model_3.pkl', scaler_filename='scaler.pkl'):
    try:
        joblib.dump(knn_model, model_filename)
        joblib.dump(scaler, scaler_filename)
        print(f"Model and scaler saved successfully.")
    except Exception as e:
        print(f"Error saving model and scaler: {e}")


def main():
    folder_paths = [
        'dataset/extracted_without_augmentation/Plant_leave_diseases_dataset_without_augmentation/Apple___Apple_scab',
        'dataset/extracted_without_augmentation/Plant_leave_diseases_dataset_without_augmentation/Apple___healthy',
        'dataset/extracted_without_augmentation/Plant_leave_diseases_dataset_without_augmentation/Corn___healthy',
        'dataset/extracted_without_augmentation/Plant_leave_diseases_dataset_without_augmentation/Corn___Common_rust',
        'dataset/extracted_without_augmentation/Plant_leave_diseases_dataset_without_augmentation/Pepper,_bell___Bacterial_spot',
        'dataset/extracted_without_augmentation/Plant_leave_diseases_dataset_without_augmentation/Pepper,_bell___healthy',
        'dataset/extracted_without_augmentation/Plant_leave_diseases_dataset_without_augmentation/Potato___Early_blight',
        'dataset/extracted_without_augmentation/Plant_leave_diseases_dataset_without_augmentation/Potato___healthy',
        'dataset/extracted_without_augmentation/Plant_leave_diseases_dataset_without_augmentation/Tomato___healthy',
        'dataset/extracted_without_augmentation/Plant_leave_diseases_dataset_without_augmentation/Tomato___Late_blight',
        'dataset/extracted_without_augmentation/Plant_leave_diseases_dataset_without_augmentation/Grape___Black_rot',
        'dataset/extracted_without_augmentation/Plant_leave_diseases_dataset_without_augmentation/Grape___healthy',

        'dataset/extracted_with_augmentation/Plant_leave_diseases_dataset_with_augmentation/Apple___Apple_scab',
        'dataset/extracted_with_augmentation/Plant_leave_diseases_dataset_with_augmentation/Apple___healthy',
        'dataset/extracted_with_augmentation/Plant_leave_diseases_dataset_with_augmentation/Corn___healthy',
        'dataset/extracted_with_augmentation/Plant_leave_diseases_dataset_with_augmentation/Corn___Common_rust',
        'dataset/extracted_with_augmentation/Plant_leave_diseases_dataset_with_augmentation/Pepper,_bell___Bacterial_spot',
        'dataset/extracted_with_augmentation/Plant_leave_diseases_dataset_with_augmentation/Potato___healthy',
        'dataset/extracted_with_augmentation/Plant_leave_diseases_dataset_with_augmentation/Potato___Early_blight',
        'dataset/extracted_with_augmentation/Plant_leave_diseases_dataset_with_augmentation/Pepper,_bell___healthy',
        'dataset/extracted_with_augmentation/Plant_leave_diseases_dataset_with_augmentation/Tomato___healthy',
        'dataset/extracted_with_augmentation/Plant_leave_diseases_dataset_with_augmentation/Tomato___Late_blight',
        'dataset/extracted_with_augmentation/Plant_leave_diseases_dataset_with_augmentation/Grape___Black_rot',
        'dataset/extracted_with_augmentation/Plant_leave_diseases_dataset_with_augmentation/Grape___healthy'

    ]

    all_features, all_labels = load_all_features_and_labels(folder_paths)

    if not all_features:
        print("No features extracted. Check the dataset and feature extraction process.")
        return

    # train the knn model and get the knn model and the scaler model
    knn_model, scaler= train_knn_model( all_features, all_labels)

    # Save the trained model and scaler
    save_model_and_scaler(knn_model, scaler)



# Execute the main function
if __name__ == "__main__":
    main()
