# Hand Gesture Recognition

This is a full-stack application that uses machine learning to recognize hand gestures in real-time from a webcam or a static image. The frontend is built with **React, TypeScript, and Vite**, and the backend is a **Python server using FastAPI and Socket.IO**.

![Screenshot](backend/Screenshot%202025-08-09%20152034.png)

## Features

-   **Real-time Gesture Recognition**: Uses a webcam to capture video and recognize hand gestures live in the browser.
-   **Image-Based Recognition**: Upload an image to detect and classify the hand gesture within it.
-   **Dual Hand Support**: Capable of recognizing gestures from one or two hands simultaneously.
-   **Efficient ML Model**: Utilizes a lightweight TensorFlow Lite model (`.tflite`) for fast inference, suitable for running on standard hardware.
-   **WebSocket Communication**: Real-time communication between the frontend and backend is handled efficiently with Socket.IO.

## Tech Stack

| Component | Technology                                                              |
| :-------- | :---------------------------------------------------------------------- |
| **Frontend**  | React, TypeScript, Vite, Tailwind CSS, shadcn/ui, Socket.IO Client      |
| **Backend**   | Python, FastAPI, Uvicorn, Socket.IO, TensorFlow Lite                  |
| **ML**        | MediaPipe (for landmark detection), TensorFlow Lite (for inference)     |
| **Deployment**| Render (Backend), GitHub Pages / Vercel / Netlify (Frontend)          |

---

## Local Development Setup

Follow these steps to run the application on your local machine.

### 1. Backend Setup (Python)

First, set up and run the Python server.

```bash
# Navigate to the backend directory
cd backend

# Create and activate a Python virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install the required Python packages
pip install -r requirements.txt

# Run the FastAPI server
uvicorn main:socket_app --reload
```

The backend server will start and be accessible at `http://localhost:8000`.

### 2. Frontend Setup (React)

In a separate terminal, set up and run the React frontend.

```bash
# Navigate to the frontend directory
cd frontend

# Install the required npm packages
npm install

# Run the Vite development server
npm run dev
```

The frontend development server will start. Open your browser and navigate to `http://localhost:3000` (or the port specified in the terminal) to use the application.

---

## Deployment

This application is ready to be deployed on modern cloud platforms.

### Backend Deployment (Render)

The backend can be deployed for free on **Render**.

1.  **Push to GitHub**: Ensure your project is a Git repository and pushed to GitHub.
2.  **Create a Web Service on Render**:
    *   Sign up on [Render](https://render.com) and create a **New Web Service**.
    *   Connect your GitHub repository.
    *   **Important Settings**:
        *   **Root Directory**: `backend`
        *   **Environment**: `Python 3`
        *   **Build Command**: `pip install -r requirements.txt`
        *   **Start Command**: `uvicorn main:socket_app --host=0.0.0.0 --port=$PORT`
        *   **Instance Type**: `Free`
3.  **Deploy**: Create the service. Render will automatically deploy your backend. Copy the public URL it provides (e.g., `https://your-app.onrender.com`).

### Frontend Deployment

The frontend can be deployed to any static hosting service like Vercel, Netlify, or GitHub Pages.

1.  **Update Backend URL**:
    *   Before deploying, you must update the backend URL in the frontend code.
    *   In `frontend/src/components/WebcamGestureRecognition.tsx` and `frontend/src/components/ImageGestureRecognition.tsx`, replace `http://localhost:8000` with your public Render backend URL.
2.  **Deploy**:
    *   **Vercel/Netlify (Recommended)**: Connect your GitHub repository to Vercel or Netlify. They will automatically detect the Vite project and deploy it.
    *   **GitHub Pages**: You can follow the instructions in `Deployment.md` if you prefer GitHub Pages, but you may need to adjust the build configuration in `vite.config.ts`.
