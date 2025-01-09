import streamlit as st
import requests

# Chatbot API URL
chatbot_api_url = "http://127.0.0.1:5000/ask"

# Page title
st.set_page_config(page_title="On Prem Gen AI LLM")

# Display a title on the page
st.title("On Prem Gen AI LLM")

# Text input for the user to type their message, with placeholder text
user_input = st.text_input("", "", placeholder="Type your query...")

# Button to send the request to the chatbot
if st.button("Send"):
    if user_input:
        # Send a POST request to the chatbot API with the user input
        response = requests.post(chatbot_api_url, json={"query": user_input})  # Correct key here

        # Display the chatbot's response
        if response.status_code == 200:
            bot_reply = response.json().get("response", "No response from chatbot.")
            st.write(f"**Chatbot**: {bot_reply}")
        else:
            st.write(f"Error: Unable to connect to chatbot. Status code: {response.status_code}")
    else:
        st.write("Please enter a message.")
