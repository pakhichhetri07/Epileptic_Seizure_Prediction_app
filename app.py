import streamlit as st
import pickle
import numpy as np

# Set the page configuration
st.set_page_config(page_title="EEG Prediction App", layout="wide")

# Load the model
try:
    with open('Model.pkl', 'rb') as file:
        model = pickle.load(file)
        st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Function to make predictions
def make_prediction(features):
    # Convert the list of features into a numpy array
    input_data = np.array(features, dtype=np.float64)

    # Check for NaN or None values in the input data
    if np.isnan(input_data).any():
        st.error("Input data contains NaN values.")
        return None
    if any(f is None for f in features):
        st.error("Input data contains None values.")
        return None

    # Reshape the data for the model (1 sample, 33 features)
    input_data = input_data.reshape(1, -1)
    prediction = model.predict(input_data)
    return int(prediction[0])

# App Title
st.title("EEG Signal Classification App")

# Introduction
st.markdown("""
This app predicts whether EEG signals indicate epileptic seizures or normal conditions.
### Instructions:
1. Enter the features as a Python array with exactly 33 elements.
2. Click on the 'Make Prediction' button to get the result.
""")

# Input field for features
input_string = st.text_area(
    "Enter features as a Python array (e.g., [1.0, 0.5, 1.3, ...]):",
    value="",
    height=100
)

# Prediction button
if st.button("Make Prediction"):
    # Parse the input string into a list of floats
    if input_string:
        try:
            # Evaluate the input string safely to convert to a list
            features = eval(input_string)  # Ensure the input is a list
            
            # Check if the input is a list and contains exactly 33 features
            if not isinstance(features, list) or len(features) != 33:
                st.error("Please provide a valid list of exactly 33 features.")
            else:
                prediction = make_prediction(features)
                if prediction is not None:
                    st.success(f"**Prediction:** {prediction} (0: Normal, 1: Epileptic)")
        except (ValueError, SyntaxError):
            st.error("Please ensure the input is a valid Python array format.")
    else:
        st.warning("Please enter the features.")

# Additional Information
st.subheader("How It Works")
st.markdown("""
The model uses features extracted from EEG signals to predict whether the input data indicates normal or epileptic conditions.
- **Input:** A list of 33 numerical features.
- **Output:** A prediction label (0 for Normal, 1 for Epileptic).
""")

# Contact Information
st.sidebar.header("Contact Information")
st.sidebar.markdown("For support, reach out to [Your Email](mailto:youremail@example.com)")

# Footer
st.write("---")
st.markdown("Made with ❤️ by Pakhi")
