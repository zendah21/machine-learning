import os
import cv2
import joblib
import numpy as np
from PIL import Image
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Function to calculate LBP features
def get_pixel(img, center, x, y): #https://www.geeksforgeeks.org/create-local-binary-pattern-of-an-image-using-opencv-python/
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except IndexError:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y-1))
    val_ar.append(get_pixel(img, center, x-1, y))
    val_ar.append(get_pixel(img, center, x-1, y + 1))
    val_ar.append(get_pixel(img, center, x, y + 1))
    val_ar.append(get_pixel(img, center, x + 1, y + 1))
    val_ar.append(get_pixel(img, center, x + 1, y))
    val_ar.append(get_pixel(img, center, x + 1, y-1))
    val_ar.append(get_pixel(img, center, x, y-1))
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val

def calculate_lbp_features(image):
    # we get the rows and colums 128 x 128
    rows, cols = image.shape

    # initialize a numpy array with zeros same size as the image 128 x 128
    lbp_values = np.zeros_like(image, dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            lbp_values[i, j] = lbp_calculated_pixel(image, i, j)
    return lbp_values.flatten()

# Function to extract LBP features from an image file
def extract_lbp_features(file_path, size=(128, 128)):
    try:
        # Load the image
        image = cv2.imread(file_path)

        # Convert the image to grayscale using cv2.cvtColor
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the grayscale image
        gray_image = cv2.resize(gray_image, size)

        # Calculate LBP features using your existing LBP calculation function
        lbp_features = calculate_lbp_features(gray_image)

        return lbp_features

    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
        return None

# Function to load the dataset
def load_dataset(folder_path, label, size=(128, 128)):
    # Load images from a folder and extract features
    features = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, filename)
            lbp_features = extract_lbp_features(file_path, size) # 1-D array of lbp values (decimal)
            if lbp_features is not None:
                features.append(lbp_features)
                labels.append(label)
    return features, labels

# Function to perform grid search and train k-NN classifier
def perform_grid_search(X_train, y_train):
    # Perform grid search to find the best hyperparameters for k-NN
    # sample code using grid search in knn https://www.kaggle.com/code/melihkanbay/knn-best-parameters-gridsearchcv
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'], # a link to the part of hyper parameters https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#:~:text='uniform'%20%3A%20uniform%20weights.,neighbors%20which%20are%20further%20away.
        'metric': ['euclidean', 'manhattan', 'minkowski'] #Distance Metrics Used in Machine Learning https://www.analyticsvidhya.com/blog/2020/02/4-types-of-distance-metrics-in-machine-learning/
    }
    # create a standard knn classifier object
    knn = KNeighborsClassifier()  # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

    # doing the grid search with hyperparameters and also cross validation
    # now we have grd search object initialized with the perimeters
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy') #this is the code source  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

    # starts the grid search process. It fits the model for every combination of hyperparameters in the parameter grid using cross-validation.
    grid_search.fit(X_train, y_train)

    #After the grid search is complete, this line returns the best estimator (model) found during the search.
    # The "best" is determined based on the specified scoring metric (accuracy in this case).

    return grid_search.best_estimator_

# how to save trained model https://neptune.ai/blog/saving-trained-model-in-python
# using joblib for serialization works with pickle files https://joblib.readthedocs.io/en/latest/persistence.html#persistence
def save_model_and_scaler(scaler, knn_model, model_filename="knn_model_potato.pkl", scaler_filename="scaler.pkl"):
    # Save the trained k-NN model to a file
    joblib.dump(knn_model, model_filename)

    # Save the StandardScaler to a file
    joblib.dump(scaler, scaler_filename)
# Main function
def main():
    # Define paths to different class folders
    folder_paths = [
        'dataset/extracted_with_augmentation/Plant_leave_diseases_dataset_with_augmentation/Potato___Early_blight',
        'dataset/extracted_with_augmentation/Plant_leave_diseases_dataset_with_augmentation/Potato___healthy',
        'dataset/extracted_with_augmentation/Plant_leave_diseases_dataset_with_augmentation/Potato___Late_blight',

        'dataset/extracted_without_augmentation/Plant_leave_diseases_dataset_without_augmentation/Potato___Late_blight',
        'dataset/extracted_without_augmentation/Plant_leave_diseases_dataset_without_augmentation/Potato___healthy',
        'dataset/extracted_without_augmentation/Plant_leave_diseases_dataset_without_augmentation/Potato___Early_blight',
    ]

    all_features = []
    all_labels = []

    # Loop through each class folder to load and extract features
    total_images = 0  # Initialize total image count
    folder_number =0
    # Loop through each class folder
    for folder_path in folder_paths:
        folder_number+=1
        print(f"Processing folder {folder_number}/{len(folder_paths)}: {folder_path}")

        # Count the number of images in the current class folder
        num_images_in_folder = sum(1 for _ in os.listdir(folder_path) if _.lower().endswith(('.png', '.jpg', '.jpeg')))
        total_images += num_images_in_folder

        # Assign label 1 for 'healthy' class, and 0 otherwise
        label = 1 if "healthy" in folder_path.lower() else 0
        #get features and labels for images in folder
        features, labels = load_dataset(folder_path, label) # [4,34,122] 1

        all_features.extend(features) #  Each element is a 1D NumPy array representing the feature vector of an image after LBP feature extraction.
        all_labels.extend(labels) # Each element is an integer label (0 or 1) corresponding to the class of an images

    print(f"Total number of images: {total_images}")

    # Before normalizing features, check to see if all_features is empty
    if not all_features:
        print("No features extracted. Check the dataset and feature extraction process.")
        return

    # This block checks whether all the feature vectors in all_features have the same length.
    # Inconsistent lengths could lead to issues during normalization or training.

    feature_lengths = [len(feature) for feature in all_features] # list of lengths
    unique_lengths = set(feature_lengths) #  it creates a set containing the unique elements of that iterable.
    if len(unique_lengths) != 1:
        print(f"Inconsistent feature vector lengths: {unique_lengths}")
        return

    # Normalize the features
    #Standardization involves scaling each feature such that it has a mean of 0 and a standard deviation of 1 for each feature.
    # why we need StandardScaler https://forecastegy.com/posts/is-feature-scaling-required-for-the-knn-algorithm/
    scaler = StandardScaler()#source  https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

    #computes the mean and standard deviation necessary for scaling and then applies the transformation to the input data
    # why we need fit_transform()  The purpose of standardization is to bring all features to a similar scale,
    # making it easier to compare and interpret their importance in a machine learning model
    # https://www.analyticsvidhya.com/blog/2021/04/difference-between-fit-transform-fit_transform-methods-in-scikit-learn-with-python-code/#:~:text=The%20fit_transform()%20method%20is,()%20and%20transform()%20separately.
    all_features_normalized = scaler.fit_transform(np.array(all_features))#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

    # Random oversampling involves randomly selecting examples from the minority class, with replacement, and adding them to the training dataset. # https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/
    ros = RandomOverSampler(sampling_strategy="minority",random_state=42) # https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html
    # now we do over sampling to the minority class to have same size as the majority class
    features_resampled, labels_resampled = ros.fit_resample(all_features_normalized, all_labels)

    # Split the dataset into training and testing sets using sklearn library
    X_train, X_test, y_train, y_test = train_test_split(features_resampled, labels_resampled, test_size=0.2, random_state=42)

    # Perform grid search for hyperparameter tuning
    best_knn = perform_grid_search(X_train, y_train)

    # Train the classifier with the best parameters
    best_knn.fit(X_train, y_train)

    # Save the trained model and scaler
    save_model_and_scaler(scaler, best_knn)

    # Predict on the test data using sklearn library
    # function enables us to predict the labels of the data values on the basis of the trained model.
    # https://www.askpython.com/python/examples/python-predict-function
    y_pred = best_knn.predict(X_test)

    # Evaluate the model provided by sk learn
    # classification report https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    # confusion matrix https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    print("Classification Report (Testing Set):")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix (Testing Set):")
    print(confusion_matrix(y_test, y_pred))

    # Classification Report
    # precision: The ratio of correctly predicted positive observations to the total predicted positives.
    # recall: The ratio of correctly predicted positive observations to all observations in the actual class.
    # f1-score: The weighted average of precision and recall. It is a good way to show that a classifier has a good value for both recall and precision.
    # support: The number of actual occurrences of the class in the specified dataset.

    # Confusion matrix
    # [[True Negative   False Positive]
    # [False Negative  True Positive]]

    # True Negative (TN): Instances that were correctly predicted as the negative class.
    # False Positive (FP): Instances that were incorrectly predicted as the positive class.
    # False Negative (FN): Instances that were incorrectly predicted as the negative class.
    # True Positive (TP): Instances that were correctly predicted as the positive class.

    # Predict on the training data
    y_train_pred = best_knn.predict(X_train)

    # Evaluate the model on the training set
    print("\nTraining Set Metrics:")
    print("Classification Report (Training Set):")
    print(classification_report(y_train, y_train_pred))
    print("Confusion Matrix (Training Set):")
    print(confusion_matrix(y_train, y_train_pred))

    # Accuracy on training set
    training_accuracy = best_knn.score(X_train, y_train)
    print(f"\nTraining Accuracy: {training_accuracy:.2f}")

    # Accuracy on testing set
    testing_accuracy = best_knn.score(X_test, y_test)
    print(f"Testing Accuracy: {testing_accuracy:.2f}")


if __name__ == "__main__":
    main()
