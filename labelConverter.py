import numpy as np
import pandas as pd

# Load the data and labels from the .npy files
data = np.load('data.npy')
labels = np.load('label.npy')

# Check if the data and labels have compatible shapes
if data.shape[0] == labels.shape[0]:
    # Convert the numpy arrays to a pandas DataFrame
    df = pd.DataFrame(data)
    
    # Add the labels as a new column
    df['label'] = labels
    
    # Save the DataFrame to a .csv file
    df.to_csv('data_with_labels.csv', index=False)
else:
    print("The number of data points and labels do not match.")

print("Conversion complete. The file 'data_with_labels.csv' has been created.")
