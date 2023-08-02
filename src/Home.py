import streamlit as st
from io import StringIO
import re
import sys
from modules.history import ChatHistory
from modules.layout import Layout
from modules.sidebar import Sidebar
import modules.embedder as embedder
from  modules.chatbot import Chatbot

#To be able to update the changes made to modules in localhost (press r)
def reload_module(module_name):
    import importlib
    import sys
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    return sys.modules[module_name]

history_module = reload_module('modules.history')
layout_module = reload_module('modules.layout')
sidebar_module = reload_module('modules.sidebar')

ChatHistory = history_module.ChatHistory
Layout = layout_module.Layout
Sidebar = sidebar_module.Sidebar

st.set_page_config(layout="wide", page_icon="💬", page_title="Asistente Virtual Demo")

# Instantiate the main components
layout, sidebar = Layout(), Sidebar()

sidebar.show_options()


embedder.ingest_docs()

def setup_chatbot(model, temperature):

    chatbot = Chatbot(model, temperature)
    st.session_state["ready"] = True

    return chatbot
# Initialize chat history
history = ChatHistory()
try:
    #todo: poner modelo y temperatura como variable de entorno
    chatbot = setup_chatbot( "gpt-3.5-turbo", 0)
    st.session_state["chatbot"] = chatbot

    if st.session_state["ready"]:
        # Create containers for chat responses and user prompts
        response_container, prompt_container = st.container(), st.container()

        with prompt_container:
            # Display the prompt form
            is_ready, user_input = layout.prompt_form()

            # Initialize the chat history
            history.initialize()

            # Reset the chat history if button clicked
            if st.session_state["reset_chat"]:
                history.reset()

            if is_ready:
                # Update the chat history and display the chat messages
                history.append("user", user_input)

                old_stdout = sys.stdout
                sys.stdout = captured_output = StringIO()

                output = st.session_state["chatbot"].conversational_chat(user_input)

                sys.stdout = old_stdout

                history.append("assistant", output)

                # Clean up the agent's thoughts to remove unwanted characters
                thoughts = captured_output.getvalue()
                cleaned_thoughts = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', thoughts)
                cleaned_thoughts = re.sub(r'\[1m>', '', cleaned_thoughts)

                # Display the agent's thoughts
                with st.expander("Display the agent's thoughts"):
                    st.write(cleaned_thoughts)

        history.generate_messages(response_container)
except Exception as e:
    st.error(f"Error: {str(e)}")







