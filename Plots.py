import seaborn as sns
import matplotlib.pyplot as plt    
import numpy as np
# Create a scatter plot of actual vs. predicted values for each target
def plot_scatter(y_test, predictions, target_names):
    plt.figure(figsize=(15, 5))
    
    for i, target in enumerate(target_names):
        plt.subplot(1, 3, i + 1)
        plt.scatter(y_test[:, i], predictions[:, i], alpha=0.5)
        plt.plot([y_test[:, i].min(), y_test[:, i].max()],
                 [y_test[:, i].min(), y_test[:, i].max()], 'k--', lw=2)
        plt.xlabel(f'Actual {target}')
        plt.ylabel(f'Predicted {target}')
        plt.title(f'Actual vs. Predicted for {target}')
    
    plt.tight_layout()
    plt.show()

# Residual plot: residual = actual - predicted
def plot_residuals(y_test, predictions, target_names):
    plt.figure(figsize=(15, 5))
    
    for i, target in enumerate(target_names):
        residuals = y_test[:, i] - predictions[:, i]
        plt.subplot(1, 3, i + 1)
        sns.histplot(residuals, kde=True, bins=30)
        plt.xlabel('Residuals')
        plt.title(f'Residuals Distribution for {target}')
    
    plt.tight_layout()
    plt.show()

# Correlation matrix of actual vs. predicted
def plot_correlation_matrix(y_test, predictions, target_names):
    # Concatenate actual and predicted values for correlation matrix
    data = np.concatenate([y_test, predictions], axis=1)
    columns = [f'Actual {target}' for target in target_names] + [f'Predicted {target}' for target in target_names]
    corr_matrix = np.corrcoef(data, rowvar=False)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, xticklabels=columns, yticklabels=columns, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Actual vs. Predicted Values')
    plt.show()
