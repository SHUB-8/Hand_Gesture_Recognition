# Deployment Instructions

This document provides instructions for deploying the frontend and backend of the hand gesture recognition application.

## Frontend (GitHub Pages)

To deploy the React frontend to GitHub Pages, follow these steps:

1.  **Set the homepage in `package.json`:**

    Open `frontend/package.json` and add a `homepage` field with the URL of your GitHub Pages site. It should look like this:

    ```json
    "homepage": "https://<your-github-username>.github.io/<your-repo-name>"
    ```

2.  **Install `gh-pages`:**

    Open a terminal in the `frontend` directory and run the following command:

    ```bash
    npm install gh-pages --save-dev
    ```

3.  **Add deploy scripts to `package.json`:**

    In `frontend/package.json`, add the following scripts to the `scripts` section:

    ```json
    "predeploy": "npm run build",
    "deploy": "gh-pages -d build"
    ```

4.  **Deploy the application:**

    Run the following command in the `frontend` directory:

    ```bash
    npm run deploy
    ```

    This will build the application and push the contents of the `build` directory to a `gh-pages` branch on your GitHub repository.

5.  **Configure GitHub Pages:**

    In your GitHub repository settings, go to the "Pages" section and select the `gh-pages` branch as the source.

## Backend (Render)

To deploy the FastAPI backend to Render, follow these steps:

1.  **Create a new Web Service on Render:**

    Go to the Render dashboard and create a new Web Service. Connect it to your GitHub repository.

2.  **Configure the service:**

    *   **Name:** Give your service a name (e.g., `hand-gesture-backend`).
    *   **Root Directory:** Set the root directory to `backend`.
    *   **Environment:** Select `Python 3`.
    *   **Build Command:** Set the build command to `pip install -r requirements.txt`.
    *   **Start Command:** Set the start command to `uvicorn main:socket_app --host 0.0.0.0 --port $PORT`.

3.  **Create the service:**

    Click the "Create Web Service" button. Render will automatically build and deploy your application.

## Update Frontend API URLs

After deploying the backend, you need to update the API URLs in the frontend to point to your Render service.

1.  **Update WebSocket URL:**

    In `frontend/src/App.tsx`, change the `socket` initialization to use your Render service URL:

    ```typescript
    const socket = io('https://<your-render-service-name>.onrender.com');
    ```

2.  **Update REST API URL:**

    In `frontend/src/App.tsx`, update the `fetch` URL in the `handleImageUpload` function:

    ```typescript
    const response = await fetch(`https://<your-render-service-name>.onrender.com/predict_image?num_hands=${numHands}`, {
      // ...
    });
    ```

3.  **Redeploy the frontend:**

    After updating the URLs, redeploy the frontend to GitHub Pages using `npm run deploy`.
