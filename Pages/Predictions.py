import streamlit as st
import pickle  # or any other library you used to save the model

# Load the pre-saved machine learning model
model_path = 'Assets/models/Key_components.pickle' 
model = pickle.load(open(model_path, 'rb'))

# Define the callback function to update session state
def update_inputs(value):
    key = st.session_context.get_id()
    st.session_state.user_inputs[key] = value

# Create three columns
col1, col2, col3 = st.columns(3)

# Initialize session state
if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = {}

# Define the options for Region and Tenure
region_options = {
    'DAKAR': 'DAKAR',
    'DIOURBEL': 'DIOURBEL',
    'FATICK': 'FATICK',
    'KAFFRINE': 'KAFFRINE',
    'KAOLACK': 'KAOLACK',
    'KEDOUGOU': 'KEDOUGOU',
    'KOLDA': 'KOLDA',
    'LOUGA': 'LOUGA',
    'MATAM': 'MATAM',
    'SAINT-LOUIS': 'SAINT-LOUIS',
    'SEDHIOU': 'SEDHIOU',
    'TAMBACOUNDA': 'TAMBACOUNDA',
    'THIES': 'THIES',
    'ZIGUINCHOR': 'ZIGUINCHOR'
}

tenure_options = {
    'D 3-6 months': 'D 3-6 months',
    'E 6-9 months': 'E 6-9 months',
    'F 9-12 months': 'F 9-12 months',
    'G 12-15 months': 'G 12-15 months',
    'H 15-18 months': 'H 15-18 months',
    'I 18-21 months': 'I 18-21 months',
    'J 21-24 months': 'J 21-24 months',
    'K > 24 months': 'K > 24 months'
}

# Column 1 inputs (Region and Tenure)
with col1:
    st.header("Column 1")
    region = st.selectbox("Region", list(region_options.keys()), on_change=update_inputs)
    tenure = st.selectbox("Tenure", list(tenure_options.keys()), on_change=update_inputs)
    recharge_amount = st.text_input("Recharge Amount", key="recharge_amount")
    recharge_frequency = st.text_input("Recharge Frequency", key="recharge_frequency")
    revenue = st.text_input("Revenue", key="revenue")

# Column 2 inputs
with col2:
    st.header("Column 2")
    arpu_segment = st.text_input("ARPU Segment", key="arpu_segment")
    income_frequency = st.text_input("Income Frequency", key="income_frequency")
    data_volume = st.text_input("Data Volume", key="data_volume")
    on_net = st.text_input("On Net", key="on_net")
    orange = st.text_input("Orange", key="orange")
    tigo = st.text_input("Tigo", key="tigo")

# Column 3 inputs
with col3:
    st.header("Column 3")
    regularity = st.text_input("Regularity", key="regularity")
    top_pack = st.text_input("Top Pack", key="top_pack")
    frequency_top_pack = st.text_input("Frequency Top Pack", key="frequency_top_pack")

# Submit button
submitted = st.button("Submit")

# Display the user inputs
st.write("User Inputs:")
for key, value in st.session_state.user_inputs.items():
    st.write(f"{key}: {value}")

# Make prediction when the submit button is clicked
if submitted:
    # Verify model type
    if isinstance(model, dict):
        raise ValueError("Model loading failed. The loaded object is not a valid machine learning model.")

    # Prepare the input data for the model
    input_data = [st.session_state.user_inputs[key] for key in st.session_state.user_inputs.keys()]

    try:
        # Make the prediction
        prediction = model.predict([input_data])

        # Display the prediction result
        st.write(f"Customer Churn Prediction: {prediction[0]}")

    except Exception as e:
        st.error(f"Prediction error: {e}")




