import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
#import pyodbc

st.set_page_config(
    page_title='View Data',
    page_icon=':📊)',
    layout='wide'
)

st.title('Exploratory Analysis of Telecommunications Data (EDA)📊📝_')


# Data Loading
import pandas as pd

# Define file paths relative to the current working directory
train_file = 'Assets/Datasets/Train.csv'
test_file = 'Assets/Datasets/Test(3).csv'
submission_file = 'Assets/Datasets/SampleSubmission(1).csv'
variable_definitions_file = 'Assets/Datasets/VariableDefinitions.csv'

# Load data
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
submission = pd.read_csv(submission_file)
variable_definitions = pd.read_csv(variable_definitions_file)


#making a copy of the train data for exploration
train_copy=train_data.copy()


# Generate descriptive statistics of the churn_train DataFrame

def display_descriptive_statistics(train_copy):
    st.write("#### SAMPLE DATA📊:")
    st.write(train_copy.head(20))
    st.subheader("DESCRIPTIVE STATISTICS")
    st.write("#### Summary Statistics for Numerical Columns:")
    st.write(train_copy.describe().T)
    st.write("#### Summary Statistics for Categorical Columns:")
    st.write(train_copy.describe(include='object').T)
    
    # st.write("#### Dataset Info:")
    # st.write(train_copy.info())
# Load the dataset and display descriptive statistics
display_descriptive_statistics(train_copy)


'### **DESCRIPTION OF COLUMNS** ###'
'**The churn dataset includes 19 variables including 15 numeric variables and 04 categorical variables.**'

'**user_id:**'

'**REGION:** *The location of each client*'

'**TENURE:** *duration in the networ*'

'**MONTANT:** *top-up amount*'

'**FREQUENCE_RECH:** *number of times the customer refilled*'

'**REVENUE:** *monthly income of each client*'

'**ARPU_SEGMENT:** *income over 90 days / 3*'

'**FREQUENCE:** *number of times the client has made an income*'

'**DATA_VOLUME:** *number of connections*'

'**ON_NET:** *inter expresso call*'

'**ORANGE:** *call to orange*'

'**TIGO:** *call to Tigo*'

'**ZONE1:** *call to zones1*'

'**ZONE2:** *call to zones2*'

'**MRG:** *a client who is going*'

'**REGULARITY:** *number of times the client is active for 90 days*'

'**TOP_PACK:** *the most active packs*'

'**FREQ_TOP_PACK:** *number of times the client has activated the top pack packages*'

'**CHURN:** whether customer will leave or not(Target Variable) where 0 == No & 1 == Yes '




# # Additional analysis
# st.subheader("Additional Analysis")
# # Correlation
# correlation = train_copy.corr(numeric_only=True)
# st.write("#### Correlation Matrix:")
# fig_corr = go.Figure(data=go.Heatmap(z=correlation.values,
#                                      x=correlation.columns,
#                                      y=correlation.columns,
#                                      colorscale='Viridis'))
# st.plotly_chart(fig_corr)




