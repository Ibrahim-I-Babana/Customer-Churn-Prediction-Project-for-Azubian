import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sklearn
import os
import datetime
#import pickle  # or any other library you used to save the model

#logTransformer class
class LogTransformer():
    def __init__(self, constant=1e-5):
        self.constant = constant

    def fit(self, x, y=None):
        return self
    def transform (self,x):
        return np.log1p(x + self.constant)
    
  #Create function to load models and cache them
st.cache_resource(show_spinner='Model Loading')
def load_GB_pipeline():
    pipeline = joblib.load('./Assets/models/GradientBoosting_model.joblib')
    return pipeline


st.cache_resource(show_spinner='Model Loading')   
def load_LR_pipeline():
    pipeline = joblib.load('./Assets/models/LogisticRegression_model.joblib')
    return pipeline
  
#Create a select model function
def choose_model():

    columns1, columns2, columns3 = st.columns(3)
    
    with columns1:
            st.selectbox('Choose a Model',options=['Gradient Boosting', 'Logistic Regression'],
            key = 'choose_model')
    with columns2:
        pass

    if st.session_state['choose_model'] == 'Logistic Regression':
        pipeline =  load_GB_pipeline()
    else:
        pipeline = load_LR_pipeline()

    #Load encoder from joblib
   
    encoder = joblib.load('./Assets/models/Label_encoder.joblib')

    return pipeline, encoder
def make_predictions(pipeline, encoder):
    #Create variables for each field by extracting using the session_state
    Region = st.session_state['Region']
    Tenure = st.session_state['Tenure']
    Recharge_Amount = st.session_state['Recharge_Amount']
    Recharge_Frequency = st.session_state['Recharge_Frequency']
    Revenue = st.session_state['Revenue']
    ARPU_Segment = st.session_state['ARPU_Segment']
    Income_Frequency = st.session_state['Income_Frequency']
    Data_Volume = st.session_state['Data_Volume']
    On_Net = st.session_state['On_Net']
    Orange = st.session_state['Orange']
    Tigo = st.session_state['Tigo']
    Regularity = st.session_state['Regularity']
    Top_Pack = st.session_state['Top_Pack']
    Frequency_Top_Pack = st.session_state['Frequency_Top_Pack']


# Load the pre-saved machine learning model
#model_path = 'Assets/models/Key_components.pickle' 
#model = pickle.load(open(model_path, 'rb'))

# Define the callback function to update session state
#def update_inputs(value):
 #   key = st.session_context.get_id()
  #  st.session_state.user_inputs[key] = value

# Create three columns
    columns = ['Region', 'Tenure', 'Recharge_Amount', 'Recharge_Frequency', 'Revenue',
       'ARPU_Segment', 'Income_Frequency', 'Data_Volume', 'On_Net', 'Orange',
       'Tigo', 'Regularity', 'Top_Pack', 'Frequency_Top_Pack']

#Create a data
    data = [[Region, Tenure, Recharge_Amount, Recharge_Frequency, Revenue,
        ARPU_Segment, Income_Frequency, Data_Volume, On_Net, Orange,
        Tigo, Regularity, Top_Pack, Frequency_Top_Pack]]

#Crete a dataframe
    Customer_predict_df =pd.DataFrame(data,columns=columns)
    Customer_predict_df['Prediction Time'] = datetime.date.today()
    df =Customer_predict_df.to_csv('/Assets/Datasets/history.csv', mode = 'a', header = not os.path.exists('/Assets/Datasets/history.csv'),index=False)

 #Predict a value from Customer_Predict_df
    Predict = pipeline.predict(Customer_predict_df)
    Prediction = int(Predict[0])
    Prediction = encoder.inverse_transform(Prediction)

#Get Probabilities
    Probability = pipeline.predict_proba(Customer_predict_df)

#Update Session States
    st.session_state['Predictions'] = Prediction
    st.session_state['Probabilities'] = Probability

    return Prediction, Probability

if 'Predictions' not in st.session_state:
    st.session_state['Predictions'] = None
if 'Probabilities' not in st.session_state:
    st.session_state['Probabilities'] = None

#(['Region', 'Tenure', 'Recharge_Amount', 'Recharge_Frequency', 'Revenue',
       #'ARPU_Segment', 'Income_Frequency', 'Data_Volume', 'On_Net', 'Orange',
       #'Tigo', 'Regularity', 'Top_Pack', 'Frequency_Top_Pack'],
      #dtype='object')
#Define functions that make predictions
#Create columns for prediction page 
def predict_page():
    pipeline, encoder = choose_model()
    
    with st.form('input_variables'):
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
        
        # Create two rows
        row1, row2 = st.columns(2)

        # Row 1 inputs (Demographic Information and Service and Preferences)
        with row1:
            with st.expander("#### Demographic Information🧑‍🦰👩‍🦰🧔‍🦰"):
                region = st.selectbox("Region", list(region_options.keys()), key="region")
                tenure = st.selectbox("Tenure", list(tenure_options.keys()), key="tenure")
                income_frequency = st.number_input("Income Frequency",  min_value=1, max_value=92, key="income_frequency")

            with st.expander("#### Service and Preferences📱💵📶📦"):
                revenue = st.number_input("Revenue", min_value=1, max_value=532178, key="revenue")
                arpu_segment = st.number_input("ARPU Segment", min_value=0, max_value=177393, key="arpu_segment")
                top_pack = st.text_input("Top Pack", key="top_pack")
                frequency_top_pack = st.number_input("Frequency Top Pack", min_value=0, max_value=625, key="frequency_top_pack")

        # Row 2 inputs (Usage Patterns)
        with row2:
            with st.expander("#### Usage Patterns💰🔄📶📞⏰📊"):
                recharge_amount = st.number_input("Recharge Amount",  min_value=20, max_value=470000, key="recharge_amount")
                recharge_frequency = st.number_input("Recharge Frequency", min_value=1, max_value=132, key="recharge_frequency")
                data_volume = st.number_input("Data Volume", min_value=0, max_value=1702310, key="data_volume")
                on_net = st.number_input("On Net", min_value=0, max_value=50810, key="on_net")
                orange = st.number_input("Orange", min_value=0, max_value=12041, key="orange")
                tigo = st.number_input("Tigo", min_value=0, max_value=4174, key="tigo")
                regularity = st.number_input("Regularity", min_value=1, max_value=63, key="regularity")

        # Submit button
        submitted = st.form_submit_button('Submit')

        if submitted:
            make_predictions(pipeline, encoder)


if __name__ == "__main__":
    st.markdown("# 📈Make a Prediction")
    predict_page()

if 'Prediction' in st.session_state and 'Probabilities' in st.session_state:
    pred = st.session_state['Prediction']
    prob = st.session_state['Probabilities'][0][1] * 100  # Assuming the probability is stored in the first element of the list
    statement = f"<h2><strong>The churn status of this customer is {pred} at a probability rate of {round(prob, 1)}%.</strong></h2>"
    st.markdown(statement, unsafe_allow_html=True)

st.write(st.session_state)

#     # Prepare the input data for the model
#     input_data = [st.session_state.user_inputs[key] for key in st.session_state.user_inputs.keys()]

#     try:
#         # Make the prediction
#         prediction = model.predict([input_data])

#         # Display the prediction result
#         st.write(f"Customer Churn Prediction: {prediction[0]}")

#     except Exception as e:
#         st.error(f"Prediction error: {e}")




