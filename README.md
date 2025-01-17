# Iris Classification using QDA

This project implements a Quadratic Discriminant Analysis (QDA) classifier to predict iris flower varieties based on their measurements. The model uses standardized features and evaluates performance using accuracy and log loss metrics.

## Project Overview

The project analyzes the classic Iris dataset, which contains measurements for three different varieties of iris flowers:
- Setosa
- Versicolor
- Virginica

For each flower, we have four measurements:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

## Technical Implementation

### Data Processing
- The dataset is split into 70% training and 30% testing sets
- Features are standardized using StandardScaler to ensure all measurements are on the same scale
- Target variable is the iris variety

### Model
- Algorithm: Quadratic Discriminant Analysis (QDA)
- QDA was chosen because it can capture non-linear relationships between features
- The model assumes each class has its own covariance matrix

### Evaluation Metrics
- Accuracy: Measures the proportion of correct predictions
- Log Loss: Measures the quality of predictions, taking into account the uncertainty

## Requirements

```
- Python 3.x
- scikit-learn
- pandas
- numpy
```

## File Structure

```
project/
│
├── iris.csv          # Input dataset
├── model.py          # Main script containing the QDA implementation
└── README.md         # This file
```

## Usage

1. Install the required packages:
```bash
pip install scikit-learn pandas numpy
```

2. Make sure your iris.csv file is in the same directory as the script

3. Run the script:
```bash
python main.py
```

## Output

The script will print:
- Accuracy score (between 0 and 1)
- Log loss value (lower is better)

## How It Works

1. **Data Loading**: The script starts by reading the iris.csv file using pandas

2. **Data Preparation**:
   - Features (X) are separated from the target variable (y)
   - Data is split into training and testing sets
   - Features are standardized to have zero mean and unit variance

3. **Model Training**:
   - QDA model is initialized and trained on the standardized training data
   - The model learns the parameters for each class

4. **Prediction**:
   - Model makes predictions on the test set
   - Both class predictions and probability estimates are generated

5. **Evaluation**:
   - Accuracy is calculated by comparing predicted classes with actual classes
   - Log loss is calculated using prediction probabilities

## Notes

- The random_state parameter is set to 42 for reproducibility
- Feature standardization is important for QDA to work effectively
- The model assumes features within each class follow a Gaussian distribution
