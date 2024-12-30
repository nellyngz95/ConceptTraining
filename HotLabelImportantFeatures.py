import numpy as np
import pandas as pd

# Load the datasets
df = pd.read_csv('DatasetFeatures.csv')  # Full dataset with all samples and features
dfeat = pd.read_csv('OneHotFeatures.csv')  # Dataset containing labels and important features

# Extract all possible feature names from the full dataset, excluding the label column
feature_names = [col for col in df.columns if col != 'Label']

# Initialize a zero-filled DataFrame for one-hot encoding
one_hot_df = pd.DataFrame(0, index=dfeat['Label'].unique(), columns=feature_names)

# Populate the one-hot encoded DataFrame based on important features
for _, row in dfeat.iterrows():
    label = row['Label']  # Get the current label
    # Extract the features marked as important for this label
    important_features = [
        row[feature] for feature in ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'] 
        if feature in row and row[feature] in feature_names
    ]
    # Set the corresponding feature columns to 1 for this label
    one_hot_df.loc[label, important_features] = 1

# Now, encode the entire dataset
encoded_samples = []

for _, row in df.iterrows():
    label = row['Label']  # Get the label of the current sample
    # Retrieve the one-hot encoded features for this label
    encoded_features = one_hot_df.loc[label]
    # Append the encoded features to the result list
    encoded_samples.append(encoded_features.values)

# Convert the list of encoded samples to a DataFrame
encoded_dataset = pd.DataFrame(encoded_samples, columns=feature_names)

# Add the labels back to the encoded dataset
encoded_dataset['Label'] = df['Label'].values

# Save the fully encoded dataset to a CSV file
encoded_dataset.to_csv('FullyEncodedDataset.csv', index=False)

# Display the encoded dataset
print(encoded_dataset.head())

