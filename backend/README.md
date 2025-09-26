# Backend: Hand Gesture Recognition API

This directory contains the backend server for the Hand Gesture Recognition application. It's built with **Python** using the **FastAPI** framework and **Socket.IO** for real-time WebSocket communication.

## Tech Stack

-   **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
-   **WebSocket**: [python-socketio](https://python-socketio.readthedocs.io/en/latest/)
-   **Server**: [Uvicorn](https://www.uvicorn.org/)
-   **Machine Learning**: [TensorFlow Lite](https://www.tensorflow.org/lite) for running the gesture classification model.
-   **Data Handling**: [NumPy](https://numpy.org/)

---

## How It Works

The server exposes a WebSocket endpoint and a REST endpoint.

1.  **WebSocket (`/socket.io`)**:
    -   The frontend sends a stream of hand landmark coordinates (features) to the `features` (single hand) or `features_dual` (two hands) events.
    -   The server receives these features, normalizes them, and feeds them into the pre-trained TensorFlow Lite model (`.tflite`).
    -   The model's prediction (the recognized gesture) is sent back to the client via a `prediction` event.

2.  **REST Endpoint (`/predict_image`)**:
    -   The frontend sends hand landmark features extracted from a static image via a POST request.
    -   The server processes the features and returns a JSON response containing the predicted gesture.

## Local Setup

1.  **Create Virtual Environment**:
    ```bash
    python -m venv venv
    ```
2.  **Activate Environment**:
    -   Windows: `.\venv\Scripts\activate`
    -   macOS/Linux: `source venv/bin/activate`
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run Server**:
    ```bash
    uvicorn main:socket_app --reload
    ```
    The server will be available at `http://localhost:8000`.
