import streamlit as st

st.title("Hello World")
st.write("Welcome to my first streamlit project")

with st.form(key='my_form'):
    name = st.text_input("Enter your name")
    age = st.number_input("Enter your age")
    email = st.text_input("Enter your email")
    password = st.text_input("Enter your password")
    submit_button = st.form_submit_button("Submit")
if submit_button:
    st.write(f"Hello {name}! Your age is {age}.")
    st.write(f"Your email is {email}.")
    st.write(f"Your password is {password}.")
    st.balloons()

