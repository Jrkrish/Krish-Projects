# Setting up the Environment
# 1.Create new conda environment:
conda create -n personal_code_assistant python=3.11
# 2.Activate the environment:
conda activate personal_code_assistant
# 3.Install all the required packages:
pip install -r requirements.txt

# Customizing and testing the model

# Go to models directory, download the model in it and then go back to the main directory of the project:
cd models
curl -O https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q4_K_M.gguf
cd ..

# Create Modelfile:
FROM codellama:7b-instruct-q4_K_M

# Define the system prompt
SYSTEM """You are an intelligent assistant - TypeBot. You excel at providing clear, insightful, and well-rounded answers to a wide range of questions.Your goal is to engage in thoughtful, helpful conversations that enrich users’ knowledge across different areas of life."""
# Set the template for instruction formatting
TEMPLATE "[INST] <<SYS>> {{ .System }} <</SYS>> {{ .Prompt }} [/INST]"

# Set parameters
PARAMETER rope_frequency_base 1e+06
PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"
PARAMETER stop "<<SYS>>"
PARAMETER stop "<</SYS>>"

# Run the command:
ollama create MODEL --file Modelfile
ollama pull codellama:7b-instruct-q4_K_M

# Run the Application:
streamlit run app.py