import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import io
import requests
import os
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

# Create a directory for model offloading
if not os.path.exists("model_cache"):
    os.makedirs("model_cache")

# Load the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "Jaykrish123/Llama-2-7b-cricket-chat-jkfinetune"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configure model loading with proper offloading
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder="model_cache",
            low_cpu_mem_usage=True
        )
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Function to generate cricket insights
def generate_cricket_insights(prompt, delay=0.05):
    try:
        model, tokenizer = load_model_and_tokenizer()
        if model is None or tokenizer is None:
            return ["Error loading model"], "Error loading model"
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        
        with torch.no_grad():
            response = model.generate(
                input_ids,
                max_length=600,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(response[0], skip_special_tokens=True)
        if generated_text.endswith('</s>'):
            generated_text = generated_text[:-len('</s>')]

        insights = [line.strip() for line in generated_text.split('\n') if line.strip()]
        return insights, generated_text
    except Exception as e:
        return [f"Error generating response: {str(e)}"], str(e)

# Function to load Lottie animation
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def main():
    st.set_page_config(page_title="🏏 CricketBot: Ultimate Cricket Assistant", layout="wide")

    # Add a loading state indicator
    with st.spinner("Initializing CricketBot... This might take a few minutes on first run."):
        # Initialize model loading in the background
        model_load_state = st.empty()
        model_load_state.info("Loading AI model... This will take a few minutes on first run.")
        load_model_and_tokenizer()
        model_load_state.empty()

    # Custom CSS with cricket-themed colors
    st.markdown('''
    <style>
        .stApp {
            background-color: #1a472a;
            color: #ffffff;
        }
        .stButton>button {
            background-color: #c4272f;
            color: white;
            border-radius: 20px;
            border: none;
            padding: 10px 20px;
            font-weight: bold;
        }
        .stTextInput>div>div>input {
            background-color: #2d4a1e;
            color: #ffffff;
            border-radius: 20px;
            border: 1px solid #c4272f;
            padding: 10px 20px;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .user-message {
            background-color: #2d4a1e;
            margin-left: 2rem;
        }
        .bot-message {
            background-color: #3d5a2e;
            margin-right: 2rem;
        }
    </style>
    ''', unsafe_allow_html=True)

    # Main content
    st.title("🏏 CricketBot: Ultimate Cricket Assistant")

    # Navigation
    selected = option_menu(
        menu_title=None,
        options=["Chat", "Stats Dashboard", "Player Analysis", "Match Predictions"],
        icons=["chat-dots", "graph-up", "person", "trophy"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )

    if selected == "Chat":
        st.header("💬 Chat with CricketBot")
        
        # Initialize session state for chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Chat input
        user_input = st.text_input("Ask me anything about cricket - stats, players, matches, or strategies!")
        
        if user_input:
            with st.spinner("Thinking..."):
                insights, full_response = generate_cricket_insights(user_input)
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("bot", full_response))
            
            # Display chat history with custom styling
            for role, message in st.session_state.chat_history:
                if role == "user":
                    st.markdown(f'<div class="chat-message user-message">You: {message}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message bot-message">CricketBot: {message}</div>', unsafe_allow_html=True)

    elif selected == "Stats Dashboard":
        st.header("📊 Cricket Statistics Dashboard")
        st.info("Coming soon: Interactive cricket statistics and visualizations")

    elif selected == "Player Analysis":
        st.header("🏃 Player Analysis")
        st.info("Coming soon: Deep dive into player statistics and performance analysis")

    elif selected == "Match Predictions":
        st.header("🎯 Match Predictions")
        st.info("Coming soon: AI-powered match outcome predictions")

    # Footer
    st.markdown("---")
    st.markdown("🏏 CricketBot - Powered by AI for Cricket Enthusiasts")
    st.markdown("Stay updated with the latest in cricket analytics!")

if __name__ == "__main__":
    main()