from fastapi import FastAPI, Body
import joblib
import boto3
import os

S3_BUCKET='cxue-mlops'
S3_KEY='model/model.pkl'
LOCAL_MODLE_PATH="models/model.pkl"

app=FastAPI()

def download_mode():
    if not os.path.exists(LOCAL_MODLE_PATH):
        os.makedirs("models", exist_ok=True)
        s3=boto3.client("s3")
        s3.download_file(S3_BUCKET, S3_KEY, LOCAL_MODLE_PATH)

# Download model ONCE at startup
download_mode()

try: 
    model, scaler = joblib.load(LOCAL_MODLE_PATH)
    print("\n-- get model --\n")
except:
    print("\n-- failed get model --\n")

@app.post("/predict")
# Use Body in FastAPI UI to receive input data. The input format looks like [[5]]
def predict(new_data: list=Body(...)):
    try:
        new_data_scaled=scaler.transform(new_data)
        predication_salary=model.predict(new_data_scaled)
        salary_str=f"${predication_salary[0]:,.2f}"
        print(f"\nsalary: ${salary_str}")
        return {"salary": salary_str}
    except:
        print("\n catch error in predict")

# predict([[5]])
