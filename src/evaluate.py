import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import json

df=pd.read_csv("data/processed/clean.csv")
x=df[['YearsExperience']] # Feature (Must be a 2D array/DataFrame)
y=df['Salary'] # Target (What we want to predict)

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale the data
scaler=StandardScaler()
scaler.set_output(transform="pandas") # This tells the scaler to return a Pandas DataFrame
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

print(f"\nFeature name: {X_train_scaled.columns}")

# Evaluate
model, loaded_scaler = joblib.load("models/model.pkl")
y_pred=model.predict(X_test_scaled)
score=r2_score(y_test, y_pred)
mean=mean_squared_error(y_test, y_pred)
print(f"\nModel Accuracy (R2 Score): {score:.2f}")
print(f"\nAverage Error (MSE): {mean:.2f}")

metrics={"r2_score": round(score, 2),
         "mean_squared_error": round(mean, 2)
         }

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)


print("\nModel evaluated and save to metrics file")
