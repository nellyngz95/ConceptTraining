import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from importlib import reload
import Plots
reload(Plots)
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
       'Energy', 'PitchSalience', 'EffectiveDuration', 'Decrease', 'Intensity',
       'DynComplex', 'LDB', 'Frequencies', 'Amplitudes', 'Mean', 'Median',
       'Variance', 'MaxToTotal', 'MinToTotal', 'TCToTotal', 'FlatnessSFX',
       'InstantPower', 'logAttackTime', 'attackStart', 'attackStop', 'Spread',
       'Skewness', 'Kurtosis', '0', '1', '2', '3', '4', '5', '6', '7', '8',
       '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
       '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32',
       '33', '34', '35', '36', '37', '38', '39', 'SpecComplex', 'RollOff',
       'StrongPeak', 'Label']]
y=df[['Loudness', 'RMS', 'spectralflux', 'Centroid', 'HighFrequency', 'ZCR']]
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
print(predictions)

predictions = np.array(predictions)
y_test = np.array(y_test)


# Define the target names
target_names = ['PitchSalience', 'HighFrequency', 'ZCR']

# Call the plotting functions
plot_scatter(y_test, predictions, target_names)
plot_residuals(y_test, predictions, target_names)
plot_correlation_matrix(y_test, predictions, target_names)
