import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Data preparation

# Load the data
train_data = pd.read_csv('mnist_train.csv', header=None, nrows=20000)
test_data = pd.read_csv('mnist_test.csv', header=None, nrows=20000)

# Separate features and labels
X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values

X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

# Define the classes to keep
classes_to_keep = [2, 3, 8, 9]

# Filter the training data
train_mask = np.isin(y_train, classes_to_keep)
X_train_filtered = X_train[train_mask]
y_train_filtered = y_train[train_mask]

# Filter the test data
test_mask = np.isin(y_test, classes_to_keep)
X_test_filtered = X_test[test_mask]
y_test_filtered = y_test[test_mask]

# Normalize the data
X_train_filtered = X_train_filtered / 255.0
X_test_filtered = X_test_filtered / 255.0

# Step 2: Train a 4-class SVM with Linear Kernel

# Define the SVM model with linear kernel
linear_svm = SVC(kernel='linear')

# Define the hyperparameters grid
param_grid = {'C': [0.1, 1, 10]}

# Use GridSearchCV for hyperparameter tuning
grid_search_linear = GridSearchCV(linear_svm, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search_linear.fit(X_train_filtered, y_train_filtered)

# Get the best model
best_linear_svm = grid_search_linear.best_estimator_

# Evaluate the model
train_accuracy_linear = best_linear_svm.score(X_train_filtered, y_train_filtered)
test_accuracy_linear = best_linear_svm.score(X_test_filtered, y_test_filtered)

print(f"Training accuracy (Linear SVM): {train_accuracy_linear}")
print(f"Test accuracy (Linear SVM): {test_accuracy_linear}")

# Step 3: Train a 4-class SVM with Non-Linear Kernel

# Define the SVM model with RBF kernel
rbf_svm = SVC(kernel='rbf')

# Define the hyperparameters grid
param_grid_rbf = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1]}

# Use GridSearchCV for hyperparameter tuning
grid_search_rbf = GridSearchCV(rbf_svm, param_grid_rbf, cv=3, n_jobs=-1, verbose=2)
grid_search_rbf.fit(X_train_filtered, y_train_filtered)

# Get the best model
best_rbf_svm = grid_search_rbf.best_estimator_

# Evaluate the model
train_accuracy_rbf = best_rbf_svm.score(X_train_filtered, y_train_filtered)
test_accuracy_rbf = best_rbf_svm.score(X_test_filtered, y_test_filtered)

print(f"Training accuracy (RBF SVM): {train_accuracy_rbf}")
print(f"Test accuracy (RBF SVM): {test_accuracy_rbf}")

# Get the indices of the support vectors
support_vector_indices = best_rbf_svm.support_

# Get the corresponding support vectors and their labels
support_vectors = best_rbf_svm.support_vectors_
support_vector_labels = y_train_filtered[support_vector_indices]

# Function to plot support vectors of a given class
def plot_support_vectors(class_label, support_vectors, support_vector_labels, n=5):
    class_support_vectors = support_vectors[support_vector_labels == class_label]
    plt.figure(figsize=(10, 2))
    for i in range(min(n, len(class_support_vectors))):
        plt.subplot(1, n, i + 1)
        plt.imshow(class_support_vectors[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.suptitle(f'Support Vectors for Class {class_label}')
    plt.show()

# Function to plot random samples of a given class
def plot_random_samples(class_label, X, y, n=5):
    class_samples = X[y == class_label]
    plt.figure(figsize=(10, 2))
    for i in range(min(n, len(class_samples))):
        plt.subplot(1, n, i + 1)
        plt.imshow(class_samples[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.suptitle(f'Random Samples for Class {class_label}')
    plt.show()

# Plot support vectors and random samples for each class
for label in classes_to_keep:
    plot_support_vectors(label, support_vectors, support_vector_labels)
    plot_random_samples(label, X_train_filtered, y_train_filtered)
