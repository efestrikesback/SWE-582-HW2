import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load data
train_data = pd.read_csv('mnist_train.csv', header=None)
test_data = pd.read_csv('mnist_test.csv', header=None)

# Filter data for digits 2, 3, 8, and 9
train_data = train_data[train_data[0].isin([2, 3, 8, 9])]
test_data = test_data[test_data[0].isin([2, 3, 8, 9])]

# Sample a subset for faster computation
train_data = train_data.sample(frac=0.2, random_state=42)
test_data = test_data.sample(frac=0.2, random_state=42)

# Separate features and labels
X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

# Train SVM with linear kernel
linear_svm = SVC(kernel='linear', C=1.0)
linear_svm.fit(X_train, y_train)

# Predict and evaluate
y_train_pred = linear_svm.predict(X_train)
y_test_pred = linear_svm.predict(X_test)

train_accuracy_linear = accuracy_score(y_train, y_train_pred)
test_accuracy_linear = accuracy_score(y_test, y_test_pred)

print(f'Training Accuracy (Linear Kernel): {train_accuracy_linear}')
print(f'Test Accuracy (Linear Kernel): {test_accuracy_linear}')

# Train SVM with RBF kernel using GridSearchCV for quick hyperparameter tuning
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
rbf_svm = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=3, n_jobs=-1)
rbf_svm.fit(X_train, y_train)

# Predict and evaluate
y_train_pred_rbf = rbf_svm.predict(X_train)
y_test_pred_rbf = rbf_svm.predict(X_test)

train_accuracy_rbf = accuracy_score(y_train, y_train_pred_rbf)
test_accuracy_rbf = accuracy_score(y_test, y_test_pred_rbf)

print(f'Training Accuracy (RBF Kernel): {train_accuracy_rbf}')
print(f'Test Accuracy (RBF Kernel): {test_accuracy_rbf}')

# Get support vectors
support_vectors = rbf_svm.best_estimator_.support_vectors_
support_vector_indices = rbf_svm.best_estimator_.support_

# Inspect support vectors
def plot_digits(data):
    fig, axes = plt.subplots(1, 10, figsize=(10, 1))
    for i, ax in enumerate(axes):
        if i < len(data):
            ax.imshow(data[i].reshape(28, 28), cmap='gray')
            ax.axis('off')
    plt.show()

print("Support Vectors:")
plot_digits(support_vectors[:10])

# Compare with random samples from the same class
random_samples = X_train[np.random.choice(len(X_train), 10, replace=False)]
print("Random Samples:")
plot_digits(random_samples)
