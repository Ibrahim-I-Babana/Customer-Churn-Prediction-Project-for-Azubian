# import streamlit as st
# import streamlit_authenticator as stauth
# import yaml
# from streamlit_authenticator import Authenticate
# from yaml.loader import SafeLoader
# from utils import column_1, column_2


# st.set_page_config(
#     page_title='Home',
#     page_icon=':)',
#     layout= 'wide'
# )
# st.title("👋 Welcome cherished user to Churn Prediction App")

# st.image("./Images/churn.png")
# with open('./config.yaml') as file:
#     config = yaml.load(file, Loader=SafeLoader)


# authenticator = stauth.Authenticate(
#     config['credentials'],
#     config['cookie']['name'],
#     config['cookie']['key'],
#     config['cookie']['expiry_days'],
#     config['preauthorized']
# )


# name, authentication_status, usernames = authenticator.login(location='sidebar')


# # utils.py

# def column_1():
#     """
#     Function to define the contents of column 1.
#     """
#     # Add your content for column 1 here
#     st.write("column 1.")

# def column_2():
#     """
#     Function to define the contents of column 2.
#     """
#     # Add your content for column 2 here
#     st.write("column 2.")



# if st.session_state["authentication_status"]:
#     authenticator.logout(location='sidebar', key='logout-button')
#     col1, col2 = st.columns(2)
#     with col1:
#         st.write("column1")
#         column_1
#     with col2:
#         st.write('### How to run application')
#         st.code('''
#         #activate virtual environment
#         env/scripts/activate
#         streamlit run 1_🏠_Home.py
#         ''')
#         column_2
#         st.link_button('Repository on GitHub', url='https://github.com/Ibrahim-I-Babana/Customer-Churn-Prediction-Project-for-Azubian.git', type='primary')

# elif st.session_state["authentication_status"] is False:
#     st.error('Username/password is incorrect')
# elif st.session_state["authentication_status"] is None:
#     st.info('Enter username and password to use the app.')
#     st.code("""
#             Test Account
#             Username: shix
#             Password: wanjiku123""")


#  # Contact Information
# with st.expander("click here to find links to the project"):
#     st.page_link("https://www.linkedin.com/in/babana-issahak-ibrahim/", label="Linkedin", icon="🌐")
#     st.page_link("https://medium.com/@lostangelib80/", label="Medium", icon="✍")
#     st.page_link("https://github.com/Ibrahim-I-Babana/Customer-Churn-Prediction-Project-for-Azubian.git", label="Github", icon="🏞️")



import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

def main():
    st.set_page_config(
        page_title="African Telecom Churn Predictor",
        page_icon=":phone:",
        layout="wide"
    )

    st.markdown('# 📞 Welcome to Telecom Churn Predictor')
    st.write("This application predicts the churn rate of customers in a telecom company.")
    st.markdown("### Please use username: Team_Hafnium and password: guest005 to access this application")
    st.write('## Key Features:')
    st.write("- Explore the factors contributing to churn rate on the **Data**📊 page.")
    st.write("- Predict churn rate using machine learning algorithms.🤖")
    st.write("- View historical churn rate data.📈")

    st.write('## How to Use:')
    st.write("- Go to the **Predict**🔮 page to make predictions.")
    st.write("- Go to the **History**📚 page to view historical churn rate data.")

    st.write("Use the sidebar⬅️ to navigate between different pages and to log in🔑.")

    # Check if user is logged in
    if 'authentication_status' not in st.session_state:
        st.session_state.authentication_status = False

    if not st.session_state["authentication_status"]:
        st.sidebar.markdown("**Guest Login Details**")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        login_button = st.sidebar.button("Login")

        if login_button:
            if username == 'Team_Hafnium' and password == 'guest005':
                st.session_state["authentication_status"] = True
            else:
                st.warning("❌ Incorrect username or password. Please try again.")

if __name__ == "__main__":
    main()