import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
 
 
st.set_page_config(
    page_title='View Data',
    page_icon='📈',
    layout='wide'
)
 
st.title(' :bar_chart: My Customer Churn Dashboard')
 
df = pd.read_csv('Assets\Datasets\Train.csv')
# Define a function to display visualizations
def display_visualizations(df):
    st.subheader('EDA Visualizations')
 
    # Set style
    sns.set_style('whitegrid')
 
    # Create columns layout
    col1, col2 = st.columns(2)
 
    # Chart for REGION distribution
    with col1:
        st.write("### REGION Distribution")
        region_counts = df['REGION'].value_counts()
        fig_region = plt.figure(figsize=(10, 6))
        region_plot = sns.barplot(y=region_counts.index, x=region_counts.values, palette='pastel')
        region_plot.set_title('REGION Distribution')
        region_plot.set_xlabel('REGION')
        region_plot.set_ylabel('Count')
        st.pyplot(fig_region)
 
    # Chart for TENURE distribution
    with col2:
        st.write("### TENURE Distribution")
        tenure_counts = df['TENURE'].value_counts()
        fig_tenure = plt.figure(figsize=(10, 6))
        tenure_plot = sns.barplot(x=tenure_counts.index, y=tenure_counts.values, palette='pastel')
        tenure_plot.set_title('TENURE Distribution')
        tenure_plot.set_xlabel('TENURE')
        tenure_plot.set_ylabel('Count')
        st.pyplot(fig_tenure)
 
    # Display the heatmap using Streamlit
    with col1:
       st.write('### Correlation Matrix Heatmap')
       numerical_df = df.select_dtypes(include=['int64', 'float64'])
       correlation = numerical_df.corr()
       plt.figure(figsize=(10, 8))
       sns.heatmap(data=correlation, annot=True, cmap='coolwarm')
       plt.title('Correlation Heatmap')
       st.pyplot(plt)
   
    with col2:
          st.write('### Statistical Summary for numerical columns')
     
 
# Assuming df is your DataFrame
          description = df.describe().T
 
# Display the result in Streamlit
          st.write(description)
 
   
def display_kpi_dashboard(df):
    st.subheader('Kpi Visualizations')
 
    # Display in two columns
    col1, col2 = st.columns(2)
 
    with col1:
        st.write("### TENURE vs. CHURN")
        tenure_churn_counts = df.groupby('TENURE')['CHURN'].value_counts().unstack()
        fig, ax = plt.subplots()
        tenure_churn_counts.plot(kind='bar', stacked=False, ax=ax)
        ax.set_xlabel('TENURE')
        ax.set_ylabel('Count')
        ax.set_title('TENURE vs. CHURN')
        ax.grid(False)  # Remove grid lines
        st.pyplot(fig)
 
    with col2:
        st.write("### Churn Percentage")
       # Create the pie chart
        piechart = px.pie(df, names='CHURN', color_discrete_sequence=['skyblue', 'pink'], width=300, height=300)
 
# Display the pie chart
        st.plotly_chart(piechart)
 
   
   
    with col1:
        st.write("### Revenue vs. Top up amount ")
        # Create the scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='MONTANT', y='REVENUE', data=df)
        ax.set_title('Scatter Plot of Top-up Amount vs REVENUE')
        ax.set_xlabel('Top-up Amount')
        ax.set_ylabel('REVENUE')
        ax.grid(False)
        st.pyplot(fig)
 
    with col2:
        st.write("### Mean Call Usage Metrics")
 
    # Separate churned and non-churned customers
        churned_customers = df[df['CHURN'] == 1]
        non_churned_customers = df[df['CHURN'] == 0]
 
    # Calculate the mean call usage metrics for churned customers
        churned_mean_metrics = churned_customers[['ON_NET', 'ORANGE', 'TIGO']].mean()
 
    # Calculate the mean call usage metrics for non-churned customers
        non_churned_mean_metrics = non_churned_customers[['ON_NET', 'ORANGE', 'TIGO']].mean()
 
    # Plotting
        fig, ax = plt.subplots(figsize=(8, 6))
 
    # Bar width
        bar_width = 0.35
 
    # Index for x-axis
        ind = range(len(churned_mean_metrics))
 
    # Plotting the mean call usage metrics for churned customers
        ax.bar(ind, churned_mean_metrics, width=bar_width, label='Churned', color='red')
 
    # Plotting the mean call usage metrics for non-churned customers
        ax.bar([x + bar_width for x in ind], non_churned_mean_metrics, width=bar_width, label='Non-Churned', color='blue')
 
    # Adding labels and title
        ax.set_xlabel('Call Usage Metrics')
        ax.set_ylabel('Mean')
        ax.set_title('Mean Call Usage Metrics for Churned and Non-Churned Customers')
        ax.set_xticks([x + bar_width/2 for x in ind])
        ax.set_xticklabels(churned_mean_metrics.index)
        ax.legend()
        ax.grid(False)
    # Showing plot
        st.pyplot(fig)
 
        # Dashboard menu at the top
dashboard_option = st.sidebar.radio("Select Dashboard", ["EDA Dashboard", "KPI Dashboard"])
 
# Content of the selected dashboard
if dashboard_option == "EDA Dashboard":
    st.write("This is the EDA Dashboard")
    display_visualizations(df)
elif dashboard_option == "KPI Dashboard":
    st.write("This is the KPI Dashboard")
    display_kpi_dashboard(df)  