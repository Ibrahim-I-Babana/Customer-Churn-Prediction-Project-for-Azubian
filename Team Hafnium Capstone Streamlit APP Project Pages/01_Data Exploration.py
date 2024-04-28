import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
#import pyodbc

st.set_page_config(
    page_title='View Data',
    page_icon=':)',
    layout='wide'
)

st.title('Exploratory Analysis of Telecommunications Data (EDA)')


# Data Loading
import pandas as pd

# Define file paths relative to the current working directory
train_file = 'Assets/Datasets/Train(1).csv'
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
    st.subheader("Descriptive Statistics")
    st.write("#### Summary Statistics:")
    st.write(train_copy.describe().T)
    st.write("#### Summary Statistics for Categorical Columns:")
    st.write(train_copy.describe(include='object').T)
    st.write("#### Sample Data:")
    st.write(train_copy.head())
    st.write("#### Dataset Info:")
    st.write(train_copy.info())



# Load the dataset and display descriptive statistics
#query = "SELECT * FROM dbo.LP2_Telco_churn_first_3000"
#df_concat = query_database(query)
display_descriptive_statistics(train_copy)

# Additional analysis
st.subheader("Additional Analysis")
# Correlation
correlation = train_copy.corr(numeric_only=True)
st.write("#### Correlation Matrix:")
fig_corr = go.Figure(data=go.Heatmap(z=correlation.values,
                                     x=correlation.columns,
                                     y=correlation.columns,
                                     colorscale='Viridis'))
st.plotly_chart(fig_corr)

# # Churn counts by Internet Service
# #Churn counts by Internet Service
# churn_counts = df_concat.groupby(['InternetService', 'Churn']).size().unstack()
# st.write("#### Churn Counts by Internet Service:")
# st.write(churn_counts)

# churn_counts = df_concat.groupby(['InternetService', 'Churn']).size().unstack()
# st.write("#### Churn Counts by Internet Service:")
# fig_churn_counts = px.bar(churn_counts, x=churn_counts.index, y=churn_counts.columns,
#                           title="Churn Counts by Internet Service", barmode='group')
# st.plotly_chart(fig_churn_counts)


