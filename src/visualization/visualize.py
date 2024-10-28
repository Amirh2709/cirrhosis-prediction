# src/visualization/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    @staticmethod
    def plot_correlation_matrix(data):
        """Plot correlation matrix."""
        plt.figure(figsize=(25, 10))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5, fmt='0.1f')
        plt.title('Correlation Matrix')
        plt.show()

    @staticmethod
    def plot_feature_importance(features, importances):
        """Plot feature importances."""
        plt.figure(figsize=(12, 8))
        plt.barh(features, importances, color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importances')
        plt.gca().invert_yaxis()
        plt.show()