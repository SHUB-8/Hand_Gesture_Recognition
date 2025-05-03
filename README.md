
# Hand Gesture Recognition Project

This project enables real-time hand gesture recognition using **MediaPipe** and **TensorFlow**, with a **Flask** or **FastAPI** backend. The system supports both **webcam-based live detection** and **image-based recognition**.

---

## Prerequisites

- **Python 3.11.x** (Recommended: 3.11.9 for full compatibility with all dependencies)

---

## ⚙Setting Up Python 3.11.x

If you don’t already have Python 3.11 installed, follow one of the methods below:

### Option 1: Using `pyenv` (Recommended)

1. Install `pyenv`:
   Follow the instructions from the [official pyenv GitHub page](https://github.com/pyenv/pyenv).

2. Install and set Python 3.11.9:

   ```bash
   pyenv install 3.11.9
   pyenv global 3.11.9
   ```

### Option 2: Using Conda

1. Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

2. Create and activate a new Python 3.11 environment:

   ```bash
   conda create -n gesture_recognition_env python=3.11.9
   conda activate gesture_recognition_env
   ```

---

## Setting Up the Project

Once Python 3.11.x is ready, follow these steps:

### 1. Clone the repository:

```bash
git clone https://github.com/SHUB-8/PBL-SE.git
cd PBL-SE
```

### 2. Create a virtual environment:

```bash
python -m venv venv
```

### 3. Activate the virtual environment:

- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```
- On Windows:
  ```bash
  venv\Scripts\activate
  ```

### 4. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

After installing the dependencies, run the application using one of the following:

### Flask Version

```bash
flask run
```

- Open in browser: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### FastAPI Version (Recommended for real-time webcam + image recognition)

```bash
uvicorn app:app --reload
```

- Open in browser: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## Testing the Application

Once the server is running, open the web app in your browser. The app will use your webcam to perform **live gesture recognition** in real time.

---

## Project Dependencies

All required Python packages are listed in `requirements.txt`. Key packages include:

```txt
Flask==2.2.5
opencv-python==4.8.0.76
numpy==1.24.4
tensorflow==2.15.0
mediapipe==0.10.9
scipy==1.10.1
```

---

## 🛠Troubleshooting

- Make sure you are using **Python 3.11.x** as specified.
- Always activate your virtual environment before installing dependencies.
- If dependencies fail, re-run:

  ```bash
  pip install -r requirements.txt
  ```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
