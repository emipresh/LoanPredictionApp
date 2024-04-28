import streamlit as st
import pandas as pd
import joblib
import warnings 
warnings.filterwarnings('ignore')

data = pd.read_csv('loan_approval_dataset.csv')

st.markdown("<h1 style = 'color: #38419D; text-align: center; font-size: 60px; font-family: Georgia'>LOAN PREDICTOR APP</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #B30000; text-align: center; font-family: italic'>Built By Eme Ita</h4>", unsafe_allow_html = True)

st.markdown("<br>", unsafe_allow_html=True)

# #add image
st.image('loanimage.png', caption ='Built by Eme Ita')

st.markdown("<h2 style = 'color: #132043; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)


st.markdown("<p>Banks and financial institutions receive numerous loan applications from customers seeking financial assistance for various purposes such as purchasing a home, starting a business, or funding education. However, approving loans without proper assessment of creditworthiness can lead to financial losses due to defaults.In today's financial landscape, access to credit plays a crucial role in fulfilling various personal and business needs. However, traditional lending institutions often face challenges in efficiently assessing the creditworthiness of loan applicants, leading to delays in loan approval and high rates of default. To address these challenges, there is a growing need for an automated loan prediction system that can accurately assess the risk associated with loan applicants and streamline the lending process.The objective of this project is to develop a loan prediction app that leverages machine learning techniques to predict the likelihood of loan approval or rejection for loan applicants based on their financial and personal attributes. The app will provide borrowers with timely feedback on their loan applications, helping them make informed decisions and improving the efficiency of the lending process for financial institutions.</p>", unsafe_allow_html = True)

st.sidebar.image('loanuser2.png', caption = 'Welcome Godstreasure')

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.header('Project Data')
st.dataframe(data, use_container_width = True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)


st.sidebar.subheader('User Input Variables')

sel_cols = [' loan_amount', ' income_annum',  ' cibil_score', ' residential_assets_value', ' commercial_assets_value',
            ' luxury_assets_value', ' no_of_dependents', ' loan_status']


loan_amt = st.sidebar.number_input('Loan Amount', data[' loan_amount'].min(), data[' loan_amount'].max())
income = st.sidebar.number_input('Income Annum', data[' income_annum'].min(), data[' income_annum'].max())
cibil  = st.sidebar.number_input('Cibil Score', data[' cibil_score'].min(), data[' cibil_score'].max())
residential = st.sidebar.number_input('Residential Asset Value', data[' residential_assets_value'].min(), data[' residential_assets_value'].max())
commercial = st.sidebar.number_input('Commercial Asset Value', data[' commercial_assets_value'].min(), data[' commercial_assets_value'].max())
luxury = st.sidebar.number_input('Luxury Asset Value', data[' luxury_assets_value'].min(), data[' luxury_assets_value'].max())
dependents = st.sidebar.number_input('No of Dependents', data[' no_of_dependents'].min(), data[' no_of_dependents'].max())


#users input
input_var = pd.DataFrame()
input_var[' loan_amount'] = [loan_amt]
input_var[' income_annum'] = [income]
input_var[' cibil_score'] = [cibil]
input_var[' residential_assets_value'] = [residential]
input_var[' commercial_assets_value'] = [commercial]
input_var[' luxury_assets_value'] = [luxury]
input_var[' no_of_dependents'] = [dependents]

# in a situation where the loanamount was scaled you should save it to a new variable 
loan_amount = int(input_var[' loan_amount'].values[0])

st.markdown("<br>", unsafe_allow_html= True)
st.divider()
st.subheader('Users Inputs')
st.dataframe(input_var, use_container_width = True)

# import the transformers
commercial_trans = joblib.load(' commercial_assets_value_scaler.pkl')
income_trans = joblib.load(' income_annum_scaler.pkl')
loan_trans = joblib.load(' loan_amount_scaler.pkl')
luxury_trans = joblib.load(' luxury_assets_value_scaler.pkl')
residential_trans = joblib.load(' residential_assets_value_scaler.pkl')

# transform the users input with the imported scalers
input_var[' commercial_assets_value'] = commercial_trans.transform(input_var[[' commercial_assets_value']])
input_var[' income_annum'] = income_trans.transform(input_var[[' income_annum']])
input_var[' loan_amount'] = loan_trans.transform(input_var[[' loan_amount']])
input_var[' luxury_assets_value'] = luxury_trans.transform(input_var[[' luxury_assets_value']])
input_var[' residential_assets_value'] = residential_trans.transform(input_var[[' residential_assets_value']])

#st.header('Transformed Input Variable') 
#st.dataframe(input_var, use_container_width = True)

# st.dataframe(input_var)
model = joblib.load('LoanpredictionModel.pkl')
predict = model.predict(input_var)

if st.button('Check Your Loan Status'):
    if predict[0] == 0:
        st.error(f"Unfortunately...Your Loan of {loan_amount} dollar has been rejectected, please try again next time")
        st.image('loanrejected.png', width = 300)
    else:
        st.success(f"Congratulations... Your loan of {loan_amount} dollar has been approved. Please proceed to any of our offices to process your loan")
        st.image('approved 2.png', width = 300)
        st.balloons()

# # Define valid username and password
# VALID_USERNAME = "godstreasure"
# VALID_PASSWORD = "6172839405"

# # Function to authenticate users
# def authenticate(username, password):
#     return username == VALID_USERNAME and password == VALID_PASSWORD

# # Streamlit app layout
# def main():

#     # Sidebar for login form
#     st.sidebar.header("Login")
#     username_input = st.sidebar.text_input("Username")
#     password_input = st.sidebar.text_input("Password", type="password")

#     # Check if login button is clicked
#     if st.sidebar.button("Login"):
#         if authenticate(username_input, password_input):
#             st.success("Logged in as {}".format(username_input))
#             # You can proceed to show the main content of the app here
#         else:
#             st.error("Invalid username or password")

# if __name__ == "__main__":
#     main()
