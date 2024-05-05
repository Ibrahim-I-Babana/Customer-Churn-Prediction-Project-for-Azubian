
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

    st.markdown('# 🏠📞 Welcome to Telecom Churn Predictor🌐')
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
