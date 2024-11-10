Acknowledgments
This project was developed by our dedicated team despite facing limited resources. We took the challenge as an opportunity to demonstrate creativity and technical prowess. Special thanks to the open-source community for making valuable tools and resources accessible.

Also thanks to the Flaunch Intern for the great accompanied journey.

🏏 CricketBot: The Ultimate Cricket Assistant
CricketBot is an AI-powered virtual assistant designed to provide cricket enthusiasts with insights, stats, player analysis, and match predictions. Powered by state-of-the-art language models, CricketBot delivers fast and insightful answers to all cricket-related questions.

Features
Chat Module: Interact with the bot and ask cricket-related questions.
Stats Dashboard: A comprehensive view of cricket statistics (upcoming).
Player Analysis: Dive into player stats and performance analysis (upcoming).
Match Predictions: AI-based match outcome predictions (upcoming).

Our Approach
This project was built with limited resources and minimal constraints. Despite these challenges, our team strived to create a powerful, user-friendly platform. By leveraging innovative techniques and optimized resource management, CricketBot is designed to provide the best possible experience even in constrained environments.

Installation and Setup Instructions
Requirements
Python 3.8 or higher
GPU acceleration (optional but recommended for better performance)
Internet connection for loading the model and dependencies

Create a virtual environment:
python -m venv venv

Activate the virtual environment:
On Windows:
.\venv\Scripts\activate
On macOS/Linux:
source venv/bin/activate

Install the required dependencies: Install all necessary libraries using the requirements.txt file:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app1.py
Open the app in your browser: After running the command, Streamlit will automatically open the app in your default web browser. If it doesn't, navigate to http://localhost:8501/ manually.

File Structure
.
├── app.py                # Main application file
├── requirements.txt      # Python dependencies
├── model_cache/          # Folder for offloading models (auto-created)
├── README.md             # Project documentation

Notes:
Model: The application uses the Llama-2 model, fine-tuned on cricket-related data, hosted on Hugging Face.
GPU Recommendation: For optimal performance, it is highly recommended to run the application on a system with a GPU.