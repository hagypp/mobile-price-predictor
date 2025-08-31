import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score
import kagglehub

# Load the model
model = load('best_model.joblib')
print("Loaded Model Type:", type(model))

# Load and preprocess data
data_path = kagglehub.dataset_download("iabhishekofficial/mobile-price-classification")
df_csv = os.path.join(data_path, "train.csv")
df = pd.read_csv(df_csv)
df['screen_size'] = df['sc_h'] * df['sc_w']
df['screen_area'] = df['px_height'] * df['px_width']
df.drop(['sc_h', 'sc_w', 'px_height', 'px_width', 'pc'], axis=1, inplace=True)

# Split data
from sklearn.model_selection import train_test_split
X = df.drop('price_range', axis=1)
y = df['price_range']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predict on test set
predictions = model.predict(X_test)
price_ranges = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Very High'}

# Create results DataFrame
results = []
for i in range(len(X_test)):
    actual = price_ranges.get(y_test.iloc[i], 'Unknown')
    predicted = price_ranges.get(predictions[i], 'Unknown')
    results.append({
        'Sample Index': i,
        'Actual': actual,
        'Predicted': predicted,
        'Correct': actual == predicted
    })

results_df = pd.DataFrame(results)
print("\nTest Set Results:")
print(results_df)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"\nTest Set Accuracy: {accuracy:.4f}")

# Show sample data
print("\nSample Test Data (RAM, Battery, Actual/Predicted Price Range):")
sample_data = X_test[['ram', 'battery_power']].copy()
sample_data['Actual Price Range'] = [price_ranges.get(y, 'Unknown') for y in y_test]
sample_data['Predicted Price Range'] = [price_ranges.get(p, 'Unknown') for p in predictions]
print(sample_data.head(10))

# Check label distribution
print("\nPrice Range Distribution in Training Data:")
print(df['price_range'].value_counts().sort_index())

# Feature importances (if applicable)
if hasattr(model, 'feature_importances_'):
    print("\nFeature Importances:")
    for name, imp in zip(X_test.columns, model.feature_importances_):
        print(f"{name}: {imp:.4f}")