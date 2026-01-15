import pandas as pd
import joblib
import boto3
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

S3_BUCKET="cxue-mlops"
S3_KEY="model/model.pkl"

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

#mlflow.set_tracking_uri("http://127.0.0.1:5000")
                        
with mlflow.start_run(): 

    print("\nStart training....")
    # Train the model
    model=LinearRegression()
    model.fit(X_train_scaled, y_train)

    print("\nStart Evaluate....")
    # Evaluate
    y_pred=model.predict(X_test_scaled)
    print(f"\nModel Accuracy (R2 Score): {r2_score(y_test, y_pred):.2f}")
    print(f"\nAverage Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")

    print("\nStart MLflow log....")
    # MLflow remember the results
    mlflow.log_metric("r2", r2_score(y_test, y_pred))
    mlflow.sklearn.log_model(model, "linear-regression-model")

    print("\nModel and metrics logged to MLflow!")

    # Save everything (modle and scaler) to a file (using joblib), so you can use this later
    joblib.dump((model, scaler), "models/model.pkl")    

print("\nModel trained")

# Predict for a new person (e.g. 5 year experrinece)
# Load and unpack in one line
loaded_model, loaded_scaler = joblib.load("models/model.pkl")

new_data=[[5]]
new_data_scaled=loaded_scaler.transform(new_data)
predication_salary=loaded_model.predict(new_data_scaled)
print(f"\nPredicted Salary for 5 years experience: ${predication_salary[0]:,.2f}")

# Upload to s3 for serving
s3=boto3.client("s3")
s3.upload_file("models/model.pkl", S3_BUCKET, S3_KEY)

print("\nModel trained and uploaded to s3")
