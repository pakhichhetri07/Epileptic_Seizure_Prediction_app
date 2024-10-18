import streamlit as st
import pickle
import numpy as np

############################
# Load the pre-trained model
############################

try:
    with open('Model.pkl', 'rb') as file:
        model = pickle.load(file)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

#############################
# Function to make prediction
#############################

def make_prediction(features):
    try:
        input_data = np.array(features, dtype=np.float64)

        # Handling Null values
        if np.isnan(input_data).any():
            st.error("Input data contains NaN values.")
            return None

        if any(f is None for f in features):
            st.error("Input data contains None values.")
            return None

        # Reshape input data
        input_data = input_data.reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)
        return int(prediction[0])

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

############################
#Build API
############################

st.title("Model Prediction App")

# take input from user
input_string = st.text_area(
    "Enter features as a Python array (e.g., [1.0, 0.5, 1.3, ...]):",
    value=""
)

if st.button("Make Prediction"):
    # Handling input in string datatype
    if input_string:
        try:
            # Convert input string into list safely
            features = eval(input_string)

            # Check if input is a list and contains 33 features
            if not isinstance(features, list) or len(features) != 33:
                st.error("Please provide a valid list of exactly 33 features.")
            else:
                prediction = make_prediction(features)
                if prediction is not None:
                    st.success(f"Prediction: {prediction}")
        except (ValueError, SyntaxError):
            st.error("Please ensure the input is a valid Python array format.")
    else:
        st.warning("Please enter the features.")

############################
#main class
############################

if __name__ == '__main__':
    st.write("Enter the features above to make a prediction.")
