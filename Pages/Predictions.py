import streamlit as st
import pickle  #library you used to save the model

# Load the pre-saved machine learning model
model = pickle.load(open('models\Key_components.pickle', 'rb'))

# Define the callback function
def update_inputs(key, value):
    st.session_state.user_inputs[key] = value

# Create three columns
col1, col2, col3 = st.columns(3)

# Initialize session state
if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = {}

# Column 1 inputs
with col1:
    st.header("Column 1")
    user_id = st.text_input("User ID", key="user_id", on_change=update_inputs, args=("user_id",))
    region = st.text_input("Region", key="region", on_change=update_inputs, args=("region",))
    tenure = st.text_input("Tenure", key="tenure", on_change=update_inputs, args=("tenure",))
    recharge_amount = st.text_input("Recharge_Amount", key="recharge_amount", on_change=update_inputs, args=("recharge_amount",))
    recharge_frequency = st.text_input("Recharge_Frequency", key="recharge_frequency", on_change=update_inputs, args=("recharge_frequency",))
    revenue = st.text_input("Revenue", key="revenue", on_change=update_inputs, args=("revenue",))

# Column 2 inputs
with col2:
    st.header("Column 2")
    arpu_segment = st.text_input("ARPU_Segment", key="arpu_segment", on_change=update_inputs, args=("arpu_segment",))
    income_frequency = st.text_input("Income_Frequency", key="income_frequency", on_change=update_inputs, args=("income_frequency",))
    data_volume = st.text_input("Data_Volume", key="data_volume", on_change=update_inputs, args=("data_volume",))
    on_net = st.text_input("On_Net", key="on_net", on_change=update_inputs, args=("on_net",))
    orange = st.text_input("Orange", key="orange", on_change=update_inputs, args=("orange",))
    tigo = st.text_input("Tigo", key="tigo", on_change=update_inputs, args=("tigo",))

# Column 3 inputs
with col3:
    st.header("Column 3")
    zone1 = st.text_input("Zone1", key="zone1", on_change=update_inputs, args=("zone1",))
    zone2 = st.text_input("Zone2", key="zone2", on_change=update_inputs, args=("zone2",))
    regularity = st.text_input("Regularity", key="regularity", on_change=update_inputs, args=("regularity",))
    top_pack = st.text_input("Top_Pack", key="top_pack", on_change=update_inputs, args=("top_pack",))
    frequency_top_pack = st.text_input("Frequency_Top_Pack", key="frequency_top_pack", on_change=update_inputs, args=("frequency_top_pack",))

# Submit button
submitted = st.button("Submit")

# Display the user inputs
st.write("User Inputs:")
for key, value in st.session_state.user_inputs.items():
    st.write(f"{key}: {value}")

# Callback function to update session state
def update_inputs(key, value):
    st.session_state.user_inputs[key] = value

# Make prediction when the submit button is clicked
if submitted:
    # Prepare the input data for the model
    input_data = [[st.session_state.user_inputs[key] for key in st.session_state.user_inputs.keys()]]

    # Make the prediction
    prediction = model.predict(input_data)

    # Display the prediction result
    st.write(f"Customer Churn Prediction: {prediction[0]}")