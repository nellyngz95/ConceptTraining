import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from importlib import reload
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import Plots
reload(Plots)
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from Plots import plot_scatter, plot_correlation_matrix, plot_residuals

# Load your data and cleaning the columns
df = pd.read_csv('TotalSound.csv')
print(df.head())
df.drop(['Unnamed: 0','c1','c2','c3','c4','c5'], axis=1, inplace=True)
print(df.head())
#Print the names of the features
print(df.columns)
#We are building one model per class in a multi output regression problem
#Split the data into training and testing sets. Determine which features are the input and which are the output
X=df[['Loudness', 'RMS', 'spectralflux', 'Centroid', 'HighFrequency', 'ZCR',
       'Energy', 'EffectiveDuration', 'Decrease', 'Intensity',
       'DynComplex', 'Frequencies', 'Amplitudes', 'Mean', 'Median',
       'Variance', 'MaxToTotal', 'MinToTotal', 'TCToTotal', 'FlatnessSFX',
       'InstantPower', 'logAttackTime', 'attackStart', 'attackStop', 'Spread',
       'Skewness', 'Kurtosis', '0', '1', '2', '3', '4', '5', '6', '7', '8',
       '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
       '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32',
       '33', '34', '35', '36', '37', '38', '39', 'SpecComplex', 'RollOff', 'Label']]
y=df[['PitchSalience', 'LDB', 'StrongPeak']]
#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
# define model
xgbreg = xgb.XGBRegressor()
#Define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(xgbreg, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = np.absolute(scores)

model = MultiOutputRegressor(xgbreg)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Print predictions for review
#print(predictions)

predictions = np.array(predictions)
y_test = np.array(y_test)


# Define the target names
target_names = ['PitchSalience', 'LDB', 'StrongPeak']

# # Call the plotting functions
# plot_scatter(y_test, predictions, target_names)
# plot_residuals(y_test, predictions, target_names)
# plot_correlation_matrix(y_test, predictions, target_names)


predictions_df = pd.DataFrame(predictions, columns=['Pred_PitchSalience', 'Pred_LDB', 'Pred_StrongPeak'])
print("Predictions",predictions_df.head())
y_test_df = pd.DataFrame(y_test, columns=['Actual_PitchSalience', 'Actual_LDB', 'Actual_StrongPeak'])
print("Actual",y_test_df.head())
error_df= pd.DataFrame(predictions-y_test, columns=['Error_PitchSalience', 'Error_LDB', 'Error_StrongPeak'])
print("Error", error_df.head())
# # Convert y_test to a DataFrame with appropriate column names–
y_test_df = pd.DataFrame(y_test, columns=['Actual_PitchSalience', 'Actual_LDB', 'Actual_StrongPeak'])


# Assuming X_test, predictions_df, error_df, y_test_df, and X are already defined
df_plot = pd.DataFrame(X_test, columns=X.columns)

df_plot=pd.merge(df_plot, predictions_df, left_index=True, right_index=True)
print(df_plot.head())
df_plot=pd.merge(df_plot, error_df, left_index=True, right_index=True)
print(df_plot.head())
df_plot=pd.merge(df_plot, y_test_df, left_index=True, right_index=True)
print(df_plot.head())


def plot_target_vs_features(target, df, features_per_subplot=8, max_labels=5):
    features = df.columns.drop(['Pred_PitchSalience', 'Pred_LDB', 'Pred_StrongPeak',
                                'Actual_PitchSalience', 'Actual_LDB', 'Actual_StrongPeak',
                                'Error_PitchSalience', 'Error_LDB', 'Error_StrongPeak', 'Label'])
    
    # Limit to the first max_labels unique labels
    unique_labels = df['Label'].unique()[:max_labels]
    for label in unique_labels:
        print(f"Label {label}: {df[df['Label'] == label].shape[0]} points")

    # Colors for each label 
    colors = plt.colormaps.get_cmap('tab20')
    color_map = {label: colors(i / max_labels) for i, label in enumerate(unique_labels)}

    norm = mcolors.Normalize(vmin=0, vmax=len(unique_labels) - 1)

    # Group features for subplots 10 per image. 
    feature_groups = [features[i:i + features_per_subplot] for i in range(0, len(features), features_per_subplot)]

    def plot_features(target, features, title_suffix):
        n_features = len(features)
        n_cols = 4
        n_rows = n_features // n_cols + (n_features % n_cols > 0)
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
        fig.suptitle(f'Scatter plots for {target} ({title_suffix})', fontsize=14)
        axs = axs.flatten()
        #PLOTTING THE SCATTER PLOTS FOR EACH FEATURE ON EVERY LABEL
        
        for i, feature in enumerate(features):
            ax = axs[i]
            print("HELP ME",unique_labels)
            for c_idx, label in enumerate(unique_labels):
                subset = df[df['Label'] == label]
                ax.scatter(np.array(subset[target]),np.array( subset[feature]), color=color_map[label], s=50, alpha=0.6)
                print(label,np.array(subset[target]),np.array( subset[feature]))
            ax.set_title(f'{target} vs. {feature}')
            ax.set_xlabel(target)
            ax.set_ylabel(feature)
        
        
        # Colorbar setup
        sm = plt.cm.ScalarMappable(cmap=colors, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axs, orientation='horizontal', fraction=0.02, pad=0.04)
        cbar.set_ticks(np.arange(len(unique_labels)))
        cbar.set_ticklabels(unique_labels)
        cbar.set_label('Labels')
        
        # Adjust layout to avoid warnings and ensure all elements fit well
        #plt.tight_layout(rect=[0, 0.05, 0.95, 0.95], h_pad=2.0, w_pad=2.0)
        plt.show()

    for i, feature_group in enumerate(feature_groups):
        plot_features(target, feature_group, f'Features {i*features_per_subplot + 1} to {i*features_per_subplot + len(feature_group)}')

# Targets to plot
targets = ['Error_PitchSalience', 'Error_LDB', 'Error_StrongPeak']

# Execute plotting function for each target
for target in targets:
    plot_target_vs_features(target, df_plot)