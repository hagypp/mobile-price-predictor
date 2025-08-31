

# Mobile Price Predictor Project

## Overview
This project predicts the price range of mobile phones (Low, Medium, High, Very High) based on their features using machine learning models. The dataset is sourced from Kaggle's [Mobile Price Classification dataset](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification). The project includes:

- A **Jupyter Notebook** (`train_model.ipynb`) that preprocesses the data, performs feature engineering, trains multiple models, and saves the best model.
- A **Streamlit app** (`app.py`) for interactive price range predictions based on user input.

The models trained include Logistic Regression, K-Nearest Neighbors (KNN), Random Forest.

## Features
- **Dataset**: Contains mobile phone features like battery power, RAM, screen size, and camera specifications, with `price_range` labels (0=Low, 1=Medium, 2=High, 3=Very High).
- **Feature Engineering**: Combines `sc_h` and `sc_w` into `screen_size`, and `px_height` and `px_width` into `screen_area`. Drops correlated features (`sc_h`, `sc_w`, `px_height`, `px_width`, `pc`).
- **Models**: Evaluates multiple classifiers using GridSearchCV to find the best model.
- **Streamlit App**: Allows users to input phone features via sliders and checkboxes to predict the price range.
- **Testing**: Validates model performance on the test set, checking for issues like reversed predictions (e.g., predicting "Very High" for low-end inputs).

## File Structure
```
mobile-price-classification/
├── train_model.ipynb       # Jupyter Notebook to train and save the model
├── app.py                  # Streamlit app for interactive predictions
├── best_model.joblib       # Saved best model (e.g., RandomForest, SVM)
└── README.md               # Project documentation
```

## Prerequisites
- Python 3.8+
- Anaconda or virtual environment
- Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn==1.7.1`
  - `xgboost`
  - `matplotlib`
  - `seaborn`
  - `streamlit`
  - `kagglehub`
- Kaggle account with API token for dataset download (see [kagglehub setup](https://github.com/Kaggle/kagglehub#authentication)).

## Setup Instructions
1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd mobile-price-predictor
   ```

2. **Set Up Environment**:
   Create and activate a Conda environment:
   ```bash
   conda create -n mobile_price python=3.8
   conda activate mobile_price
   ```

3. **Install Dependencies**:
   Install required libraries:
   ```bash
   pip install pandas numpy scikit-learn==1.7.1 xgboost matplotlib seaborn streamlit kagglehub
   ```

4. **Download Dataset**:
   The dataset is automatically downloaded via `kagglehub` in the notebook and tester scripts. Ensure internet access.


## Usage
### 1. Training the Model
- **File**: `train_model.ipynb`
- **Description**: Loads the dataset, performs feature engineering, trains multiple models (Logistic Regression, KNN, Random Forest, XGBoost, MLP, SVM), evaluates them using GridSearchCV, and saves the best model as `best_model.joblib`.
- **Steps**:
  1. Open the notebook in Jupyter:
     ```bash
     jupyter notebook train_model.ipynb
     ```
  2. Run all cells to:
     - Download and preprocess the dataset.
     - Create features (`screen_size`, `screen_area`).
     - Train and evaluate models.
     - Save the best model (`best_model.joblib`).
  3. Check outputs:
     - Model accuracies and confusion matrices.
     - Best model name and accuracy.
     - Predictions for low-end and high-end inputs.
     - Test set accuracy and sample predictions.
     - Feature importances (if applicable).

### 2. Running the Streamlit App
- **File**: `app.py`
- **Description**: A web app for interactive price range predictions using the trained model.
- **Steps**:
  1. Ensure `best_model.joblib` is in the project directory.
  2. Run the app:
     ```bash
     streamlit run app.py
     ```
  3. Open the provided URL in a browser (e.g., `http://localhost:8501`).
  4. Use sliders and checkboxes to input phone features (e.g., RAM, battery power).
  5. Click "Predict Price Range" to see the predicted price range (Low, Medium, High, Very High).


## License
This project is for educational purposes and uses the [Mobile Price Classification dataset](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification) under its respective license.

## Acknowledgments
- Dataset: [Kaggle Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)
- Libraries: scikit-learn, XGBoost, Streamlit, kagglehub

