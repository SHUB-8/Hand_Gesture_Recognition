# Hand Gesture Recognition Project

This project allows real-time hand gesture recognition using MediaPipe and TensorFlow, with a Flask (or FastAPI) backend and webcam-based real-time gesture recognition.

## Prerequisites

- **Python 3.11.x** (Recommended: 3.11.9 for full compatibility with all dependencies)
  
### Setting Up Python 3.11.x

If you don't have Python 3.11 installed, you can install it using the following methods:

#### Option 1: Using pyenv (Recommended)
1. **Install pyenv**:
   - Follow the instructions to install `pyenv` from the [official pyenv GitHub page](https://github.com/pyenv/pyenv).
   
2. **Install Python 3.11.9**:
   ```bash
   pyenv install 3.11.9
   pyenv global 3.11.9
Option 2: Using Conda
Install Miniconda/Anaconda:

Download and install from Anaconda's website.

Create a Python 3.11 environment:

bash
Copy
Edit
conda create -n gesture_recognition_env python=3.11.9
conda activate gesture_recognition_env
Setting Up the Project
Once you have Python 3.11 installed and activated, you can proceed with the project setup:

Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/gesture-recognition.git
cd gesture-recognition
Create a virtual environment:
It's recommended to use a virtual environment to avoid dependency conflicts. Run the following command:

bash
Copy
Edit
python -m venv venv
Then activate the virtual environment:

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
With the virtual environment activated, install the required dependencies from the requirements.txt file:

bash
Copy
Edit
pip install -r requirements.txt
Running the Project
After installing the dependencies, you can run the application. If you're using Flask, start the app with:

bash
Copy
Edit
flask run
If you're using FastAPI (for real-time webcam feed and image-based recognition):

bash
Copy
Edit
uvicorn app:app --reload
Testing the Application
Once the server is running, you can open the web app in your browser. The application will use your webcam for real-time gesture recognition.

Project Dependencies
The required Python packages are listed in the requirements.txt file:

txt
Copy
Edit
Flask==2.2.5
opencv-python==4.8.0.76
numpy==1.24.4
tensorflow==2.15.0
mediapipe==0.10.9
scipy==1.10.1
Troubleshooting
Python Version: If you receive errors related to the Python version, ensure you have Python 3.11.x installed as specified in the prerequisites section.

Missing Dependencies: If you run into issues with missing dependencies, make sure your virtual environment is activated before running pip install -r requirements.txt.

License
This project is licensed under the MIT License - see the LICENSE file for details.

yaml
Copy
Edit

---

### Summary of Key Points:
- Clear instructions to set up Python 3.11.
- Instructions to create and activate a virtual environment.
- Easy installation of dependencies via `requirements.txt`.
- Explanation of how to run the project (Flask or FastAPI).
  
