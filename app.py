import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.schema import AIMessage, HumanMessage

def get_realtime_response(user_prompt, model="codellama/CodeLlama-7b-Instruct-hf", **config):
    # Load the base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    inputs = tokenizer(user_prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Get the PEFT model
    peft_model = get_peft_model(base_model, lora_config)

    # Tokenize and generate
    inputs = tokenizer(user_prompt, return_tensors="pt")
    outputs = peft_model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Advanced Custom CSS with sophisticated design
custom_css = """
<style>
/* Advanced Color System */
:root {
    --color-primary: #2D5AF0;
    --color-secondary: #7D3CF0;
    --color-accent: #F0B63C;
    --color-background: #F5F7FF;
    --color-surface: #FFFFFF;
    --color-error: #F0523C;
    --color-success: #3CF094;
    
    --gradient-primary: linear-gradient(135deg, var(--color-primary), var(--color-secondary));
    --gradient-surface: linear-gradient(135deg, #FFFFFF, #F5F7FF);
    
    --shadow-sm: 0 2px 4px rgba(45, 90, 240, 0.1);
    --shadow-md: 0 4px 8px rgba(45, 90, 240, 0.15);
    --shadow-lg: 0 8px 16px rgba(45, 90, 240, 0.2);
    
    --border-radius-sm: 8px;
    --border-radius-md: 12px;
    --border-radius-lg: 20px;
}

/* Global Styles */
.stApp {
    background: var(--color-background);
    background-image: 
        radial-gradient(circle at 100% 0%, rgba(45, 90, 240, 0.1) 0%, transparent 25%),
        radial-gradient(circle at 0% 100%, rgba(125, 60, 240, 0.1) 0%, transparent 25%);
    font-family: 'Inter', system-ui, sans-serif;
}

/* Typography Enhancements */
h1 {
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.75rem;
    text-align: center;
    margin-bottom: 2rem;
    padding: 1rem;
    position: relative;
}

h1::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 100px;
    height: 4px;
    background: var(--gradient-primary);
    border-radius: 2px;
}

/* Enhanced Input Styling */
.stTextInput > div > div > input {
    background: var(--color-surface);
    border: 2px solid rgba(45, 90, 240, 0.1);
    border-radius: var(--border-radius-md);
    padding: 1rem 1.25rem;
    font-size: 1rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: var(--shadow-sm);
}

.stTextInput > div > div > input:focus {
    border-color: var(--color-primary);
    box-shadow: 0 0 0 4px rgba(45, 90, 240, 0.1);
    outline: none;
}

/* Button Refinements */
.stButton > button {
    background: var(--gradient-primary);
    color: white;
    border: none;
    border-radius: var(--border-radius-md);
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: var(--shadow-md);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

/* Sophisticated Chat Message Styling */
.chat-message {
    padding: 1.5rem;
    border-radius: var(--border-radius-lg);
    margin-bottom: 1.5rem;
    position: relative;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.chat-message::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    border-radius: var(--border-radius-lg);
    z-index: -1;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.chat-message:hover {
    transform: translateY(-2px);
}

.chat-message.user {
    background: var(--gradient-primary);
    color: white;
    margin-left: 2rem;
    box-shadow: var(--shadow-md);
}

.chat-message.assistant {
    background: var(--gradient-surface);
    border: 1px solid rgba(45, 90, 240, 0.1);
    margin-right: 2rem;
    box-shadow: var(--shadow-sm);
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: var(--color-background);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(var(--color-primary), var(--color-secondary));
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(var(--color-secondary), var(--color-primary));
}

/* Advanced Animations */
@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}

.chat-message {
    animation: slideIn 0.3s cubic-bezier(0.4, 0, 0.2, 1) forwards;
}

/* Responsive Design Enhancements */
@media (max-width: 768px) {
    h1 {
        font-size: 2rem;
    }
    
    .chat-message {
        margin-left: 1rem;
        margin-right: 1rem;
    }
}

/* Glass Morphism Effect */
.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(255, 255, 255, 0.4);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    z-index: -1;
}
</style>
"""

# app config
st.set_page_config(page_title="🤖 AI Assistant ✨", page_icon="🤖")

# Inject custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

st.title("🤖 LoRA TypeBot - Your Finetuned Assistant! ✨")

# Setting generation configuration
get_config_gen = configure_generation()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am TypeBot - Your AI Assistant! How can I help you?"),
    ]


# Display chat history with styled messages
for message in st.session_state.chat_history:
    message_type = "assistant" if isinstance(message, AIMessage) else "user"
    with st.chat_message(message_type):
        st.markdown(f'<div class="chat-message {message_type}">{message.content}</div>', unsafe_allow_html=True)

# user input
if user_query := st.chat_input("Type your message here..."):
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("human"):
        st.markdown(f'<div class="chat-message user">{user_query}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = get_realtime_response(user_prompt=user_query)
        response_placeholder.markdown(f'<div class="chat-message assistant">{full_response}</div>', unsafe_allow_html=True)
        st.session_state.chat_history.append(AIMessage(content=full_response))