# Cirrhosis Prediction Model

## Overview
This project implements various machine learning models to predict cirrhosis disease progression based on patient medical data. The models include Random Forest, XGBoost, Support Vector Machines (SVM), and Long Short-Term Memory (LSTM) networks.

## Project Structure
```plaintext
cirrhosis-prediction/
│
├── data/
│   └── raw/
│       └── cirrhosis_data.csv
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── models.py
│   └── visualization/
│       ├── __init__.py
│       └── visualize.py
│
├── requirements.txt
├── README.md
└── .gitignore 
```

- `data/`: Raw and processed data
- `models/`: Trained model files
- `notebooks/`: Jupyter notebooks for exploration
- `src/`: Source code
  - `data/`: Data loading and processing
  - `features/`: Feature engineering
  - `models/`: Model training and prediction
  - `visualization/`: Data visualization
- `tests/`: Unit tests


## Features
- Multiple model implementations (Random Forest, XGBoost, SVM, LSTM)
- Feature importance analysis
- Correlation analysis
- Comprehensive evaluation metrics
- Time series prediction capabilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Amirh2709/cirrhosis-prediction.git
cd cirrhosis-prediction
```

2. Create a virtual environment:
```bash
python -m slenv
source slenv/bin/activate  # On Windows: slenv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation
Place your dataset in the `data/raw` directory and run the data preparation script:

```python
from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer

# Load data
loader = DataLoader('data/raw/your_dataset.csv')
data = loader.load_data()

# Create features
engineer = FeatureEngineer()
features = engineer.create_features(data)
```

### Training Models
```python
from src.models.train_model import ModelTrainer

# Initialize and train a model
trainer = ModelTrainer(model_type='random_forest')
trainer.train(X_train, y_train)

# Evaluate the model
metrics = trainer.evaluate(X_test, y_test)
print(metrics)
```

### Visualization
```python
from src.visualization.visualize import Visualizer

# Plot correlation matrix
Visualizer.plot_correlation_matrix(data)

# Plot feature importance
Visualizer.plot_feature_importance(features, importances)
```

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact
Amirhossein Saberi - amir.saberi2709@gmail.com
Project Link: [https://github.com/amir/cirrhosis-prediction](https://github.com/Amirh2709/cirrhosis-prediction.git)
