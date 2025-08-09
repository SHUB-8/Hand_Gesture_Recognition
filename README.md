# Hand Gesture Recognition

This project is a real-time hand gesture recognition application built with a React frontend and a FastAPI backend.

## Project Structure

-   `frontend/`: The React application.
-   `backend/`: The FastAPI application.

## Getting Started

### Backend (FastAPI)

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    -   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    -   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the backend server:**
    ```bash
    uvicorn main:socket_app --reload
    ```
    The backend will be running at `http://localhost:8000`.

### Frontend (React)

1.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```

2.  **Install the dependencies:**
    ```bash
    npm install
    ```

3.  **Run the frontend development server:**
    ```bash
    npm start
    ```
    The frontend will be running at `http://localhost:3000`.

## Deployment

For instructions on how to deploy the application to GitHub Pages and Render, please see the `Deployment.md` file.