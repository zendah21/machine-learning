import os
import cv2
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
# Constants
IMAGE_SIZE = (128, 128)
OVERSAMPLING_STRATEGY = "minority"
RANDOM_STATE = 42
PCA_COMPONENTS = 50  # Adjust the number of PCA components
BATCH_SIZE = 500     # Adjust batch size for IncrementalPCA




def get_pixel(img, center, x, y):
    try:
        return 1 if img[x][y] >= center else 0
    except IndexError:
        return 0


def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = [
        get_pixel(img, center, x - 1, y - 1),
        get_pixel(img, center, x - 1, y),
        get_pixel(img, center, x - 1, y + 1),
        get_pixel(img, center, x, y + 1),
        get_pixel(img, center, x + 1, y + 1),
        get_pixel(img, center, x + 1, y),
        get_pixel(img, center, x + 1, y - 1),
        get_pixel(img, center, x, y - 1)
    ]
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    return sum(val_ar[i] * power_val[i] for i in range(len(val_ar)))


def plot_data_distribution(labels, title):
    plt.figure(figsize=(6, 4))
    num_bins = len(np.unique(labels))
    plt.hist(labels, bins=num_bins, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Class Label')
    plt.ylabel('Number of Samples')
    plt.xticks([0, 1], ['Unhealthy', 'Healthy'])
    plt.show()


def plot_standardization_effect(features_before, features_after):
    """
    Plot the effect of standardization on feature distributions.

    Parameters:
        features_before (list): List of original features.
        Features_after (list): List of standardized features.
        :param features_before:
        :param features_after:
    """
    # Convert lists to NumPy arrays
    features_before_array = np.array(features_before)
    features_after_array = np.array(features_after)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].scatter(features_before_array[:, 0], features_before_array[:, 1], c='b', alpha=0.5)
    axes[0].set_title('Before Standardization')
    axes[0].set_xlabel('Feature Value')
    axes[0].set_ylabel('Frequency')
    axes[1].scatter(features_after_array[:, 0], features_after_array[:, 1], c='r', alpha=0.5)
    axes[1].set_title('After Standardization')
    axes[1].set_xlabel('Feature Value')
    axes[1].set_ylabel('Frequency')
    plt.show()

def plot_ipca_effect(features_before, features_after):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].scatter(features_before[:, 0], features_before[:, 1], c='b', alpha=0.5)
    axes[0].set_title('Before IPCA')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[1].scatter(features_after[:, 0], features_after[:, 1], c='r', alpha=0.5)
    axes[1].set_title('After IPCA')
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('Principal Component 2')
    plt.show()


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


def load_all_features_and_labels(folder_paths):
    all_features = []
    all_labels = []
    total_images = 0

    for folder_path in folder_paths:
        print(f"Processing folder: {folder_path}")
        num_images_in_folder = sum(1 for _ in os.listdir(folder_path) if _.lower().endswith(('.png', '.jpg', '.jpeg')))
        total_images += num_images_in_folder
        label = 1 if "healthy" in folder_path.lower() else 0
        features, labels = load_dataset(folder_path, label)
        all_features.extend(features)
        all_labels.extend(labels)

    print(f"Total number of images: {total_images}")
    return all_features, all_labels


def plot_data_distribution_before_and_after_oversampling(features_before, labels_before, features_after, labels_after):
    # Convert to NumPy arrays if not already
    features_before_array = np.asarray(features_before)
    features_after_array = np.asarray(features_after)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot original data before oversampling
    axes[0].scatter(features_before_array[labels_before == 0, 0], features_before_array[labels_before == 0, 1],
                    c='lightblue', marker='.', alpha=0.5, label='Unhealthy (Original)')
    axes[0].scatter(features_before_array[labels_before == 1, 0], features_before_array[labels_before == 1, 1],
                    c=(1, 0.8, 0.8), marker='.', alpha=0.5, label='Healthy (Original)')

    # Plot oversampled data after SMOTE
    axes[1].scatter(features_after_array[labels_after == 0, 0], features_after_array[labels_after == 0, 1],
                    c='darkblue', marker='.', alpha=0.8, label='Unhealthy (SMOTE)')
    axes[1].scatter(features_after_array[labels_after == 1, 0], features_after_array[labels_after == 1, 1],
                    c='darkred', marker='.', alpha=0.8, label='Healthy (SMOTE)')

    axes[0].set_title('Original Data Distribution')
    axes[1].set_title('After SMOTE Distribution')

    for ax in axes:
        ax.set_xlabel('Feature Value 1')
        ax.set_ylabel('Feature Value 2')
        ax.legend()

    plt.tight_layout()
    plt.show()
def plot_standardization_effect_before_and_after(features_before, features_after):
    # Convert to NumPy arrays if not already
    features_before_array = np.asarray(features_before)
    features_after_array = np.asarray(features_after)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].scatter(features_before_array[:, 0], features_before_array[:, 1], c='darkred', marker='.', alpha=0.5)
    axes[0].set_title('Before Standardization')
    axes[0].set_xlabel('Feature Value')
    axes[0].set_ylabel('Frequency')

    axes[1].scatter(features_after_array[:, 0], features_after_array[:, 1], c='darkblue', marker='.', alpha=0.8)
    axes[1].set_title('After Standardization')
    axes[1].set_xlabel('Feature Value')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def train_knn_model(features, labels, n_neighbors=3, weights='distance', metric='euclidean', test_size=0.2,
                    cv=5, random_state=RANDOM_STATE, batch_size=BATCH_SIZE):
    # Apply SMOTE to the entire dataset
    oversampler = SMOTE(sampling_strategy=OVERSAMPLING_STRATEGY, random_state=random_state)
    oversampled_features, oversampled_labels = oversampler.fit_resample(features, labels)

    # Display the distribution of the dataset after oversampling
    plot_data_distribution_before_and_after_oversampling(features,labels,oversampled_features,oversampled_labels)

    # Standardize the features using oversampled features
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(oversampled_features)

    # Display the effect of standardization
    plot_standardization_effect_before_and_after(features, features_standardized)

    # Initialize KNN model
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)

    # Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    # Split the data into training and testing sets after oversampling
    X_train, X_test, y_train, y_test = train_test_split(features_standardized, oversampled_labels, test_size=test_size,
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

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_prediction))

    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_prediction))

    # Accuracy
    accuracy = knn_model.score(X_test_normalized, y_test)
    print(f"\nAccuracy: {accuracy:.2f}")

    # ROC Curve
    y_probabilities = knn_model.predict_proba(X_test_normalized)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def save_model_and_scaler(knn_model, scaler, model_filename='knn_model.pkl', scaler_filename='scaler.pkl'):
    try:
        joblib.dump(knn_model, model_filename)
        joblib.dump(scaler, scaler_filename)
        print(f"Model and scaler saved successfully.")
    except Exception as e:
        print(f"Error saving model and scaler: {e}")





def main():
    folder_paths = [
        'dataset/extracted_without_augmentation/Plant_leave_diseases_dataset_without_augmentation/Apple___Apple_scab',
        'dataset/extracted_without_augmentation/Plant_leave_diseases_dataset_without_augmentation/Apple___healthy'

    ]

    all_features, all_labels = load_all_features_and_labels(folder_paths)

    if not all_features:
        print("No features extracted. Check the dataset and feature extraction process.")
        return

    # train the knn model and get the knn model and the scaler model
    knn_model, scaler = train_knn_model(all_features, all_labels)

    # Save the trained model and scaler
    save_model_and_scaler(knn_model, scaler)


# Execute the main function
if __name__ == "__main__":
    main()
