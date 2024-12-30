import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import xgboost as xgb
import matplotlib as plt
import shap
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score, RepeatedKFold
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,make_scorer,mean_absolute_error,classification_report
from math import sqrt
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import shap
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('/Users/nellygarcia/Documents/ConceptLearning/DatasetFeatures.csv')

# Count occurrences of each unique label in the 'Label' column
label_counts = df['Label'].value_counts()

# Define a sorted list of labels for the legend box display order
sorted_labels = sorted(label_counts.index)

# Define a custom color palette for paired colors
custom_colors = [
    "#D4A5FF", "#7E57C2",  # Light purple, Dark purple
    "#A5D6A7", "#388E3C",  # Light green, Dark green
    "#FFCC80", "#F57C00",  # Light orange, Dark orange
    "#90CAF9", "#1976D2",  # Light blue, Dark blue
    "#FFABAB", "#D32F2F",  # Light red, Dark red
    "#FFD54F", "#FBC02D",  # Light yellow, Dark yellow
    "#B39DDB", "#512DA8",  # Light lavender, Dark lavender
    "#C5E1A5", "#689F38",  # Light lime green, Dark olive green
    "#FFAB91", "#D84315",  # Light coral, Dark coral
    "#80DEEA", "#00838F",  # Light cyan, Dark teal
    "#FFE082", "#FFA000",  # Light amber, Dark amber
    "#F8BBD0", "#C2185B",  # Light pink, Dark pink
    "#B2EBF2", "#006064",  # Light aqua, Dark aqua
    "#DCEDC8", "#33691E",  # Light moss green, Dark forest green
    "#FFCDD2", "#B71C1C",  # Light rose, Dark maroon
    "#CE93D8", "#8E24AA",  # Light magenta, Dark magenta
    "#FFF59D", "#FBC02D",  # Light golden yellow, Dark golden yellow
    "#C5CAE9", "#303F9F",  # Light periwinkle, Dark indigo
    "#FFECB3", "#FF6F00",  # Light peach, Dark peach
    "#D1C4E9", "#512DA8",  # Light violet, Dark violet
    "#B3E5FC", "#0288D1",  # Light sky blue, Dark sky blue
    "#FFCCBC", "#D84315",  # Light terracotta, Dark terracotta
    "#E6EE9C", "#AFB42B",  # Light lime, Dark olive
    "#C8E6C9", "#2E7D32",  # Light mint green, Dark emerald green
    "#FFCDD2", "#D32F2F",  # Light rose pink, Dark crimson
    "#BBDEFB", "#1976D2",  # Light azure, Dark azure
    "#D7CCC8", "#5D4037",  # Light taupe, Dark brown
    "#FFAB40", "#F57C00",  # Light tangerine, Dark tangerine
    "#DCEDC8", "#689F38",  # Light celery green, Dark olive green
    "#FFECB3", "#FFA000"   # Light honey yellow, Dark honey yellow
]

# Ensure the color palette has enough colors for all unique labels
num_labels = len(label_counts)
if num_labels > len(custom_colors):
    raise ValueError("Custom color palette has fewer colors than needed for the labels.")

# Plot the pie chart with the custom color palette
plt.figure(figsize=(10, 8))
wedges, texts = plt.pie(
    label_counts,
    labels=label_counts.index,
    autopct=None,  # Remove the percentages from the chart
    startangle=140,
    colors=custom_colors[:num_labels],  # Slice the list to match the number of labels
    labeldistance=1.2  # Move labels outside of the pie chart
)

# Add a legend with labels in sorted order and colors in 3 columns
plt.legend(
    wedges,
    sorted_labels,  # Sorted labels for the legend display
    title="Labels",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1),
    ncol=3  # Adjust as needed for better fit
)

# Add a title
plt.title('Distribution of Labels of the SynthFx 12K')
plt.savefig('/Users/nellygarcia/Documents/ConceptLearning/pie_chart.png', bbox_inches='tight')
# Display the plot
plt.show()


correlation_matrix = datas.corr()
# Set the figure size
plt.figure(figsize=(19, 10))

# Create a heatmap using the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

feature_names = [
    'Frequency1', 'Amplitude1', 'Frequency2', 'Amplitude2', 'Frequency3', 'Amplitude3',
    'Frequency4', 'Amplitude4', 'Frequency5', 'Amplitude5', 'Loudness', 'RMS',
    'SpectralFlux', 'Centroid', 'HighFrequencyContent', 'ZCR', 'Energy', 'PitchSalience',
    'EffectiveDuration', 'Decrease', 'Intensity', 'DynComplexity', 'LDB', 'CM1', 'CM2',
    'CM3', 'CM4', 'CM5', 'Mean', 'Median', 'Variance', 'InstantPower', 'Crest',
    'MaxToTotal', 'MinToTotal', 'TCToTotal', 'FlatnessSFX', 'LogAttackTime', 'AttackStart',
    'AttackStop', 'Spread', 'Skewness', 'Kurtosis', 'PitchSalience.1', 'PitchValues',
    'PitchConfidence', 'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5', 'MFCC_6',
    'MFCC_7', 'MFCC_8', 'MFCC_9', 'MFCC_10', 'MFCC_11', 'MFCC_12', 'MFCC_13', 'Label'
]

feature_names = [feature for feature in feature_names if feature != 'Label']
X = datas[feature_names]
print (X)


#XGBOOST

# Ensure feature names are unique by converting the list to a set and back to a list
feature_names = list(set(feature_names))

# Ensure that 'MFCC_13' is only appended once
if 'MFCC_13' not in feature_names:
    feature_names.append('MFCC_13')


# Now proceed with defining X and y
X = datas[feature_names]
y = datas['Label']

# Continue with the rest of your code as before
valid_labels = range(60)  # Expected labels are between 0 and 59
valid_indices = y.isin(valid_labels)
X, y = X[valid_indices], y[valid_indices]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.4, stratify=y_encoded, random_state=42)

y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.4, stratify=y_encoded, random_state=42)

# Initialize the XGBoost classifier
xgb_classifier = xgb.XGBClassifier(num_class=len(np.unique(y_encoded)), eval_metric='mlogloss')

# Fit the model to the training data
xgb_classifier.fit(X_train, y_train)

plt.figure(figsize=(10, 8))
xgb.plot_importance(xgb_classifier, importance_type="weight", max_num_features=20)
plt.title("Top 20 Features Importance - XGBoost")
plt.show()

#Random Forest

rf = RandomForestClassifier(
    max_depth=50,
    max_features=int(sqrt(X_train.shape[1])),
    min_samples_leaf=2,
    min_samples_split=2,
    n_estimators=20
)

param_grid = {
    'n_estimators': [5, 10, 20],
    'max_depth': [10, 50, 100, None],
    'max_features': ['sqrt', 'log2', None]
}


grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
rf.fit(X_train, y_train)
print("RandomForest Feature Importances:", rf.feature_importances_)

#ACCURACY:

xgb_predictions = xgb_classifier.predict(X_test)
rf_predictions = rf.predict(X_test)

# Calculate accuracies
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)

xgb_predictions = xgb_classifier.predict(X_test)
rf_predictions = rf.predict(X_test)

# Calculate F1 scores
xgb_f1_score = f1_score(y_test, xgb_predictions, average='weighted')  # Adjust average as needed
rf_f1_score = f1_score(y_test, rf_predictions, average='weighted')    # Adjust average as needed

# Display F1 scores
print("XGBoost Classifier F1 Score:", xgb_f1_score)
print("Random Forest Classifier F1 Score:", rf_f1_score)
# Display accuracies
print("XGBoost Classifier Accuracy:", xgb_accuracy)
print("Random Forest Classifier Accuracy:", rf_accuracy)

# Detailed classification reports
print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_predictions))
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_predictions))


# Get classification reports as dictionaries
xgb_report = classification_report(y_test, xgb_predictions, output_dict=True)
rf_report = classification_report(y_test, rf_predictions, output_dict=True)

# Convert classification reports to DataFrames for easier manipulation
xgb_df = pd.DataFrame(xgb_report).transpose()
rf_df = pd.DataFrame(rf_report).transpose()

# Filter only label rows (exclude 'accuracy', 'macro avg', 'weighted avg')
xgb_label_scores = xgb_df.iloc[:-3]  # Exclude summary rows
rf_label_scores = rf_df.iloc[:-3]

# Sort by F1 scores to get the bottom 10
xgb_bottom_f1 = xgb_label_scores.sort_values(by="f1-score").head(10)
rf_bottom_f1 = rf_label_scores.sort_values(by="f1-score").head(10)

# Plot the results
def plot_bottom_f1(df, model_name):
    plt.figure(figsize=(12, 6))
    plt.barh(df.index, df["f1-score"], color="skyblue")
    plt.xlabel("F1 Score")
    plt.ylabel("Labels")
    plt.title(f"Top 10 Labels with Lowest F1 Scores - {model_name}")
    plt.gca().invert_yaxis()  # Invert y-axis to show lowest score at the top
    plt.show()

# Plot for both models
plot_bottom_f1(xgb_bottom_f1, "XGBoost Classifier")
plot_bottom_f1(rf_bottom_f1, "Random Forest Classifier")

# Confusion matrix for XGBoost Classifier
xgb_conf_matrix = confusion_matrix(y_test, xgb_predictions)

# Confusion matrix for RandomForest Classifier
rf_conf_matrix = confusion_matrix(y_test, rf_predictions)

# Function to create and save confusion matrix
def create_and_save_confusion_matrix(conf_matrix, title, filename):
    fig, ax = plt.subplots(figsize=(12, 10))  # Create figure and axes explicitly
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title(title)
    fig.savefig(filename, bbox_inches='tight')  # Save using the figure object
    plt.close(fig)  # Close the figure

# Create and save the confusion matrices
create_and_save_confusion_matrix(xgb_conf_matrix, "XGBoost Classifier Confusion Matrix", "xgb_confusion_matrix.png")
create_and_save_confusion_matrix(rf_conf_matrix, "Random Forest Classifier Confusion Matrix", "rf_confusion_matrix.png")

explainer = shap.TreeExplainer(rf)  # Assuming `rf` is your random forest model
shap_values = explainer.shap_values(X_test)  # Get SHAP values for the test set

# Loop through each class (label) from 0 to 60 (or based on your unique class labels)
for i in range(61):  # Labels are from 0 to 60
    shap.summary_plot(shap_values[i], X_test, feature_names=feature_names, show=False)
    plt.title(f"Class {i} Feature Importance via SHAP")
    plt.savefig(f"/Users/nellygarcia/Documents/ConceptLearning/ShapValuesDataset/Label{i}_feature_importance.png")
    plt.close()
# If you want to plot the aggregated importance across all classes (optional)
average_shap_values = np.mean(np.abs(shap_values), axis=0)  # Average the absolute SHAP values across classes
shap.summary_plot(average_shap_values, X_test, feature_names=feature_names)


# Apply PCA to the features
pca = PCA(n_components=5)  # You can choose the number of components you want to visualize
X_pca = pca.fit_transform(X)

# Store the explained variance ratio (importance of each component)
explained_variance = pca.explained_variance_ratio_

# Create a DataFrame with PCA components
pca_df = pd.DataFrame(X_pca, columns=["Pitch Salience", "attackStart","DynComplexity","FlatnessSFX","LogAttackTime"])

# Optionally, visualize the explained variance for each component
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance)
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance')
plt.title('Explained Variance by Each Principal Component')
plt.savefig('/Users/nellygarcia/Documents/ConceptLearning/PCA/pca_explained_variance.png')
plt.close()


loadings = pca.components_.T

loadings_df = pd.DataFrame(loadings, columns=["Pitch Salience", "attackStart","DynComplexity","FlatnessSFX","LogAttackTime"], index=feature_names)

# Visualize the feature loadings
plt.figure(figsize=(10, 8))
loadings_df.plot(kind="bar", figsize=(12, 6))
plt.title('Feature Contributions to Principal Components')
plt.ylabel('Loading')
plt.xlabel('Features')
plt.savefig('/Users/nellygarcia/Documents/ConceptLearning/PCA/pca_feature_contributions.png')
