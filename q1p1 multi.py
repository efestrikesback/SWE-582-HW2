import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading

# Load data
train_data = pd.read_csv('mnist_train.csv', header=None)
test_data = pd.read_csv('mnist_test.csv', header=None)

# Filter data for digits 2, 3, 8, and 9
train_data = train_data[train_data[0].isin([2, 3, 8, 9])]
test_data = test_data[test_data[0].isin([2, 3, 8, 9])]

# Separate features and labels
X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to train SVM with grid search
def train_svm(kernel, param_grid):
    svm_model = SVC(kernel=kernel)
    grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    return kernel, train_accuracy, test_accuracy, best_model, grid_search.best_params_

# Define parameter grids for grid search
param_grids = {
    'linear': {'C': [0.1, 1, 10, 100]},
    'rbf': {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto']}
}

# Function to print execution time
def print_execution_time(start_time):
    while True:
        time.sleep(10)
        elapsed_time = time.time() - start_time
        print(f'Execution time: {elapsed_time:.2f} seconds')

# Start execution time thread
start_time = time.time()
execution_time_thread = threading.Thread(target=print_execution_time, args=(start_time,))
execution_time_thread.daemon = True
execution_time_thread.start()

# Train SVMs concurrently
kernels = ['linear', 'rbf']
results = []

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {executor.submit(train_svm, kernel, param_grids[kernel]): kernel for kernel in kernels}
    for future in as_completed(futures):
        kernel, train_accuracy, test_accuracy, model, best_params = future.result()
        results.append((kernel, train_accuracy, test_accuracy, model, best_params))
        print(f'Best Parameters for {kernel} Kernel: {best_params}')
        print(f'Training Accuracy ({kernel} Kernel): {train_accuracy}')
        print(f'Test Accuracy ({kernel} Kernel): {test_accuracy}')

# Extract the RBF SVM model for further inspection
rbf_svm = next(model for kernel, train_accuracy, test_accuracy, model, best_params in results if kernel == 'rbf')

# Get support vectors
support_vectors = rbf_svm.support_vectors_
support_vector_indices = rbf_svm.support_

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
