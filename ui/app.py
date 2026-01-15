import streamlit as st
import requests
import json

# When running locally
#API_URL="http://localhost:8000/predict"

# When running in EC2
public_ip = requests.get('https://api.ipify.org').text
print(f"Public IP: {public_ip}")
API_URL=f"http://{public_ip}:8901/predict"
print(API_URL)

st.title("Salary prediction input")

# Create a text input
user_input=st.text_input("Enter year of experience as a list:", value="[[5]]")

try:
    # Convert the string into a real python list
    input_list=json.loads(user_input)

    if st.button("Predict"):
        response=requests.post(API_URL, json=input_list)
        if response.status_code==200:
            result=response.json()
            prediction=result.get("salary", result)

            st.markdown(
                f"""
                ### Predication Result
                **salary:** `{prediction}`
                """
                )
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            st.error("Failed to get prediction from API")
except json.JSONDecodeError:
    st.error("Invalid format! Please enter a list like: [[1,2,3]]")
except requests.exceptions.ConnectionError:
    st.error("Could not connect to the API. Is FastAPI running on EC2?")
