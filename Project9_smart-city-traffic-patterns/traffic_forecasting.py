import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
import os

warnings.filterwarnings("ignore")

# Define paths
DATA_DIR = r"D:\Internship\UCT\Project9_smart-city-traffic-patterns\Project9_smart-city-traffic-patterns\smart-city-traffic-patterns"
TRAIN_PATH = os.path.join(DATA_DIR, "train_aWnotuB.csv")
TEST_PATH = os.path.join(DATA_DIR, "datasets_8494_11879_test_BdBKkAj.csv")
SUBMISSION_PATH = os.path.join(DATA_DIR, "submission.csv")

print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Convert DateTime to datetime objects
train_df['DateTime'] = pd.to_datetime(train_df['DateTime'])
test_df['DateTime'] = pd.to_datetime(test_df['DateTime'])

# 1. Basic EDA
print("Generating EDA plots...")
os.makedirs(os.path.join(DATA_DIR, "plots"), exist_ok=True)

# Traffic over time per junction
plt.figure(figsize=(15, 6))
sns.lineplot(data=train_df, x='DateTime', y='Vehicles', hue='Junction', palette='tab10')
plt.title('Traffic Volume Over Time by Junction')
plt.savefig(os.path.join(DATA_DIR, "plots", "traffic_over_time.png"))
plt.close()

# 2. Feature Engineering
def extract_features(df):
    df = df.copy()
    df['Year'] = df['DateTime'].dt.year
    df['Month'] = df['DateTime'].dt.month
    df['Day'] = df['DateTime'].dt.day
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
    
    # Simple holiday/special occasion logic (can be expanded using the `holidays` package)
    # For now, we will flag some common major dates (e.g., Dec 25, Jan 1, etc.)
    df['Is_Holiday'] = ((df['Month'] == 12) & (df['Day'] == 25) | 
                        (df['Month'] == 1) & (df['Day'] == 1)).astype(int)
    return df

print("Extracting features...")
train_features = extract_features(train_df)
test_features = extract_features(test_df)

features = ['Junction', 'Year', 'Month', 'Day', 'Hour', 'DayOfWeek', 'Is_Weekend', 'Is_Holiday']
target = 'Vehicles'

X_train = train_features[features]
y_train = train_features[target]
X_test = test_features[features]

# 3. Modeling
print("Training RandomForest Regressor...")
model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Feature Importance Plot
importances = model.feature_importances_
plt.figure(figsize=(10, 5))
sns.barplot(x=importances, y=features)
plt.title('Feature Importances')
plt.savefig(os.path.join(DATA_DIR, "plots", "feature_importance.png"))
plt.close()

# 4. Prediction
print("Generating predictions on test set...")
predictions = model.predict(X_test)
test_features['Vehicles'] = np.round(predictions).astype(int)

# Create submission
submission = test_features[['ID', 'Vehicles']]
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")
