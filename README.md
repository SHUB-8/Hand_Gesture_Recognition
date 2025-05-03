# ✋ Hand Gesture Recognition Project

This project enables real-time hand gesture recognition using **MediaPipe** and **TensorFlow**, with a **Flask** or **FastAPI** backend. The system supports both **webcam-based live detection** and **image-based recognition**.

## 🧱 Prerequisites

- **Python 3.11.x** (Recommended: **3.11.9** for full compatibility with all dependencies)

## ⚙️ Setting Up Python 3.11.x

If you don’t already have Python 3.11 installed, follow one of the methods below.

### ✅ Option 1: Using `pyenv` (Recommended)

1. Install `pyenv`:  
   Follow the instructions from the [official pyenv GitHub page](https://github.com/pyenv/pyenv).

2. Install and set Python 3.11.9:
   ```bash
   pyenv install 3.11.9
   pyenv global 3.11.9
✅ Option 2: Using Conda
Download and install Miniconda or Anaconda.

Create and activate a new Python 3.11 environment:

bash
Copy
Edit
conda create -n gesture_recognition_env python=3.11.9
conda activate gesture_recognition_env
🚀 Setting Up the Project
Once Python 3.11.x is ready, follow these steps to set up the project:

Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/gesture-recognition.git
cd gesture-recognition
Create a virtual environment:

bash
Copy
Edit
python -m venv venv
Activate the virtual environment:

On macOS/Linux:

bash
Copy
Edit
source venv/bin/activate
On Windows:

bash
Copy
Edit
venv\Scripts\activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
▶️ Running the Project
After installing the dependencies, you can run the application using one of the following:

🧪 Flask Version
bash
Copy
Edit
flask run
⚡ FastAPI Version (Recommended for real-time webcam + image recognition)
bash
Copy
Edit
uvicorn app:app --reload
🧪 Testing the Application
Once the server is running:

Open your browser at:

http://127.0.0.1:5000/ (for Flask)

http://127.0.0.1:8000/ (for FastAPI)

The app will use your webcam to perform live gesture recognition in real time.

📦 Project Dependencies
The required packages are listed in requirements.txt:

text
Copy
Edit
Flask==2.2.5
opencv-python==4.8.0.76
numpy==1.24.4
tensorflow==2.15.0
mediapipe==0.10.9
scipy==1.10.1
🛠 Troubleshooting
Python Version Errors:
Ensure you're using Python 3.11.x. Other versions may cause incompatibilities with TensorFlow or MediaPipe.

Missing Dependencies:
Ensure your virtual environment is activated before running:

bash
Copy
Edit
pip install -r requirements.txt
Reinstalling Environment (if needed):

bash
Copy
Edit
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
