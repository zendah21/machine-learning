import os
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.random_projection import GaussianRandomProjection
import matplotlib.pyplot as plt
# Constants
FEATURES_FOLDERS = [
    'Plant_leave_diseases_dataset_with_augmentation_feature_extracted',
    'Plant_leave_diseases_dataset_without_augmentation_feature_extracted'
]
OVERSAMPLING_STRATEGY = "minority"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

def load_features(folder_path, n_components=1500):
    all_features = []
    all_labels = []
    count = 0  # Initialize count of features

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pkl'):
            file_path = os.path.join(folder_path, filename)
            loaded_features = joblib.load(file_path)
            count += len(loaded_features)  # Increment count by the number of features in the file

            # Apply random projection for dimensionality reduction
            random_projection = GaussianRandomProjection(n_components=n_components, random_state=RANDOM_STATE)
            reduced_features = random_projection.fit_transform(loaded_features)

            if count == 1:
                print_feature_info(reduced_features[0], filename)

            all_features.extend(reduced_features)
            label = 0 if 'healthy' in filename.lower() else 1
            all_labels.extend([label] * len(reduced_features))

    print(f"Total number of features: {count}")

    return all_features, all_labels

def train_knn_model(X_train, y_train, n_neighbors=7, weights='distance', metric='euclidean', random_state=RANDOM_STATE):
    # Initialize KNN model
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)

    # Initialize Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)

    # Cross-validated scores
    cross_val_scores = cross_val_score(knn_model, X_train, y_train, cv=skf)

    print(f"Cross-Validation Scores: {cross_val_scores}")
    print(f"Mean Cross-Validation Score: {np.mean(cross_val_scores):.2f}")

    # Oversampling the training data using SMOTE
    smote = SMOTE(sampling_strategy=OVERSAMPLING_STRATEGY, random_state=random_state)
    X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train)
    # Visualize feature distribution before and after SMOTE
   # plot_feature_scatter(np.array(X_train), np.array(y_train), np.array(X_train_oversampled),
    #                     np.array(y_train_oversampled), title='Feature Distribution Before and After SMOTE')

    # Train the KNN model
    knn_model.fit(X_train_oversampled, y_train_oversampled)

    return knn_model


def plot_feature_scatter(X_before, y_before, X_after, y_after, title):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X_before[:, 0], X_before[:, 1], c=y_before, cmap=plt.cm.Paired, label='Before SMOTE')
    plt.title('Before SMOTE')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(X_after[:, 0], X_after[:, 1], c=y_after, cmap=plt.cm.Paired, label='After SMOTE')
    plt.title('After SMOTE')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def display_model_performance(knn_model, X_test, y_test):
    y_prediction = knn_model.predict(X_test)
    y_proba = knn_model.predict_proba(X_test)[:, 1]  # Probabilities for positive class
    print("Classification Report:")
    print(classification_report(y_test, y_prediction))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_prediction))
    accuracy = knn_model.score(X_test, y_test)
    print(f"\nAccuracy: {accuracy:.2f}")

    # Calculate ROC curve
   # fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    # Calculate AUC
    # auc = roc_auc_score(y_test, y_proba)
    # print(f"AUC: {auc:.2f}")
    #
    # # Plot ROC curve
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend(loc='lower right')
    # plt.show()

def save_model(model, filename='knn_model_binary.joblib'):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

def print_feature_info(feature, feature_name):
    # Get the size of the feature
    size = feature.nbytes  # Size in bytes

    # Calculate the storage space
    storage_space = size / (1024 * 1024)  # Convert bytes to megabytes

    print(f"Feature Information for '{feature_name}':")
    print(f"Size: {size} bytes")
    print(f"Storage Space: {storage_space:.2f} MB")
    print(f"Shape: {feature.shape}")
    print(f"Dtype: {feature.dtype}")

def main():
    all_features = []
    all_labels = []

    for folder in FEATURES_FOLDERS:
        print(f"Processing folder: {folder}")
        # Load features and labels with random projection
        features, labels = load_features(folder)  # Specify the number of components

        # Concatenate features and labels
        all_features.extend(features)
        all_labels.extend(labels)

    if len(all_features) == 0:
        print("No features extracted. Check the dataset and feature extraction process.")
        return
    
    feature_lengths = [len(feature) for feature in all_features]
    if len(set(feature_lengths)) != 1:
        print("Inconsistent feature vector lengths:", set(feature_lengths))
        return


    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=TEST_SIZE,
                                                        random_state=RANDOM_STATE)

    # Train a KNN model
    print("Training KNN model...")
    knn_model = train_knn_model(X_train, y_train)

    # Display model performance on a test set
    print("Model Performance on Test Set:")
    display_model_performance(knn_model, X_test, y_test)

    # Save the trained model
    save_model(knn_model)

if __name__ == "__main__":
    main()
