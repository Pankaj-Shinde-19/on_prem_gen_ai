import streamlit as st
import requests
from datetime import datetime

USER_AVATAR = "🧑‍💻"
BOT_AVATAR = "🤖"

st.set_page_config(page_title="On Prem Gen AI LLM",
                   page_icon="C:\\Users\\PankajShinde\\PycharmProjects\\testing-on-premise-gen-ai\\frontend\\logo.png",
                   layout="wide")

if 'session_id' not in st.session_state:
    # Initialize session state variables
    st.session_state['session_id'] = datetime.now().timestamp()
    st.session_state.messages = []

with open('C:\\Users\\PankajShinde\\PycharmProjects\\testing-on-premise-gen-ai\\frontend\\style.css') as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Function to check if user input is a greeting
def is_greeting(message):
    greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    return any(greet in message.lower() for greet in greetings)

def set_question(question):
    st.session_state["my_question"] = question

my_question = st.session_state.get("my_question", default=None)

# Display chat history
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# User input
if my_question := st.chat_input("Ask me a question"):
    set_question(None)
    st.session_state.messages.append({"role": "user", "content": my_question})

    user_message = st.chat_message("user", avatar=USER_AVATAR)
    user_message.write(f"{my_question}")

    try:
        if is_greeting(my_question):
            # Respond with a greeting if the user greeted
            assistant_response = "Greetings! I’m your on-premise assistant, here to help you with any questions from our knowledge base."
        else:
            # Make a request to the external API for non-greeting inputs
            response = requests.post(
                "http://127.0.0.1:5000/ask",
                json={"query": my_question}
            )
            response_data = response.json()

            if response.status_code == 200:
                assistant_response = response_data.get("response", "I couldn't generate a response for that question.")
            else:
                assistant_response = "Failed to retrieve data from the server."

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        assistant_message = st.chat_message("assistant", avatar=BOT_AVATAR)
        assistant_message.write(assistant_response)

    except Exception as ex:
        error_message = "An error occurred while processing the request. Please try again later."
        st.session_state.messages.append({"role": "assistant", "content": error_message})

        assistant_message = st.chat_message("assistant", avatar=BOT_AVATAR)
        assistant_message.error(error_message)
        st.write(f"Error: {ex}")
        print(f"Error: {ex}")
