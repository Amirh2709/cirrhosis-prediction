# src/data/data_loader.py
import pandas as pd


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """Load the dataset and perform initial preprocessing."""
        df = pd.read_csv(self.file_path, index_col=0)
        df['Month 1 (% Disease)'] = df['Month 1 (% Disease)'].astype(str).str.rstrip('%').astype(float) / 100
        return df