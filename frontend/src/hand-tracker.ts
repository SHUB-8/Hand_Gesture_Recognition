import { HandLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';

// --- Feature Extraction Logic ---
// The following functions process the 3D hand landmarks from MediaPipe
// into a flattened feature vector that can be fed into the classification model.

/**
 * Calculates the Euclidean distance between all pairs of landmarks.
 * @param landmarks - An array of 3D landmarks from MediaPipe.
 * @returns A 2D distance matrix.
 */
function calculateDistanceMatrix(landmarks: any[]) {
  const numLandmarks = landmarks.length;
  const distMatrix = Array(numLandmarks).fill(0).map(() => Array(numLandmarks).fill(0));

  for (let i = 0; i < numLandmarks; i++) {
    for (let j = i; j < numLandmarks; j++) {
      const dx = landmarks[i].x - landmarks[j].x;
      const dy = landmarks[i].y - landmarks[j].y;
      const dz = landmarks[i].z - landmarks[j].z;
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
      distMatrix[i][j] = distance;
      distMatrix[j][i] = distance;
    }
  }
  return distMatrix;
}

/**
 * Extracts the upper triangle of a matrix and flattens it into a 1D array.
 * This is done to create a consistent feature vector from the symmetric distance matrix.
 * @param matrix - A 2D square matrix.
 * @returns A 1D array of the upper triangle values.
 */
function getUpperTriangle(matrix: number[][]) {
  const upperTriangle: number[] = [];
  for (let i = 0; i < matrix.length; i++) {
    for (let j = i + 1; j < matrix[i].length; j++) {
      upperTriangle.push(matrix[i][j]);
    }
  }
  return upperTriangle;
}

/**
 * Extracts a feature vector from a single hand's landmarks.
 * @param landmarks - An array of 3D landmarks.
 * @returns A flattened feature vector.
 */
function extractFeatures(landmarks: any[]) {
  const distMatrix = calculateDistanceMatrix(landmarks);
  return getUpperTriangle(distMatrix);
}

/**
 * The HandTracker class encapsulates the MediaPipe HandLandmarker setup,
 * video stream handling, and the real-time prediction loop.
 */
export class HandTracker {
  private handLandmarkerVideo: HandLandmarker | undefined;
  private handLandmarkerImage: HandLandmarker | undefined;
  private videoStream: MediaStream | undefined;
  private lastVideoTime = -1;
  private animationFrameId: number | undefined;
  private canvasCtx: CanvasRenderingContext2D | null = null;
  private frameCounter = 0;
  private frameSkip = 1; // Process every frame for maximum responsiveness.

  /**
   * Initializes both the video and image HandLandmarker models from MediaPipe.
   * This is an async operation that should be called once.
   */
  async initialize() {
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm'
    );
    
    // Create the landmarker for video processing
    this.handLandmarkerVideo = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: 2,
    });

    // Create the landmarker for image processing
    this.handLandmarkerImage = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
        delegate: 'GPU',
      },
      runningMode: 'IMAGE',
      numHands: 2,
    });
  }

  /**
   * Starts the webcam, initializes the canvas, and begins the prediction loop.
   * @param videoElement - The HTMLVideoElement to stream the webcam to.
   * @param canvasElement - The HTMLCanvasElement for drawing landmarks.
   * @param callback - A function to call with the extracted features, hand count, landmarks, and handedness.
   */
  async start(videoElement: HTMLVideoElement, canvasElement: HTMLCanvasElement, callback: (features: number[], numHands: number, landmarks: any[], handedness: any[]) => void) {
    if (!this.handLandmarkerVideo) {
      console.error("HandTracker not initialized. Call initialize() first.");
      return;
    }

    this.videoStream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1920 }, height: { ideal: 1080 } } });
    videoElement.srcObject = this.videoStream;
    await videoElement.play();

    this.canvasCtx = canvasElement.getContext('2d');
    
    // Start the prediction loop.
    this.predictWebcam(videoElement, canvasElement, callback);
  }

  /**
   * Stops the webcam stream and cancels the prediction loop.
   */
  stop() {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }
    if (this.videoStream) {
      this.videoStream.getTracks().forEach(track => track.stop());
    }
  }

  /**
   * The main prediction loop. It uses requestAnimationFrame to continuously
   * process video frames.
   */
  private predictWebcam = (videoElement: HTMLVideoElement, canvasElement: HTMLCanvasElement, callback: (features: number[], numHands: number, landmarks: any[], handedness: any[]) => void) => {
    if (!this.handLandmarkerVideo || !this.canvasCtx) return;

    const nowInMs = Date.now();
    if (videoElement.currentTime !== this.lastVideoTime) {
      this.lastVideoTime = videoElement.currentTime;

      this.frameCounter++;
      if (this.frameCounter % this.frameSkip === 0) {
        const handLandmarkerResult = this.handLandmarkerVideo.detectForVideo(videoElement, nowInMs);

        this.canvasCtx.save();
        this.canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        
        if (handLandmarkerResult.landmarks && handLandmarkerResult.landmarks.length > 0) {
          // Note: Landmark drawing is disabled as per user request.
          
          if (handLandmarkerResult.landmarks.length === 1) {
            const features = extractFeatures(handLandmarkerResult.landmarks[0]);
            callback(features, 1, handLandmarkerResult.landmarks, handLandmarkerResult.handedness);
          } else {
            const features1 = extractFeatures(handLandmarkerResult.landmarks[0]);
            const features2 = extractFeatures(handLandmarkerResult.landmarks[1]);
            const combinedFeatures = [...features1, ...features2];
            callback(combinedFeatures, 2, handLandmarkerResult.landmarks, handLandmarkerResult.handedness);
          }
        } else {
          // If no hands are detected, send an empty array to clear the UI state.
          callback([], 0, [], []);
        }
        this.canvasCtx.restore();
      }
    }

    // Continue the loop.
    this.animationFrameId = requestAnimationFrame(() => this.predictWebcam(videoElement, canvasElement, callback));
  };

  /**
   * Detects hand landmarks from a single image file.
   * @param file - The image file to process.+
   * @returns A promise that resolves with the features, hand count, and landmarks.
   */
  async detectImage(file: File): Promise<{features: number[], numHands: number, landmarks: any[]} | undefined> {
    if (!this.handLandmarkerImage) {
      console.error("HandTracker not initialized. Call initialize() first.");
      return;
    }

    const image = new Image();
    image.src = URL.createObjectURL(file);
    return new Promise((resolve) => {
      image.onload = () => {
        const handLandmarkerResult = this.handLandmarkerImage!.detect(image);
        if (handLandmarkerResult.landmarks && handLandmarkerResult.landmarks.length > 0) {
          if (handLandmarkerResult.landmarks.length === 1) {
            const features = extractFeatures(handLandmarkerResult.landmarks[0]);
            resolve({ features, numHands: 1, landmarks: handLandmarkerResult.landmarks });
          } else {
            const features1 = extractFeatures(handLandmarkerResult.landmarks[0]);
            const features2 = extractFeatures(handLandmarkerResult.landmarks[1]);
            const combinedFeatures = [...features1, ...features2];
            resolve({ features: combinedFeatures, numHands: 2, landmarks: handLandmarkerResult.landmarks });
          }
        } else {
          resolve(undefined);
        }
      };
    });
  }
}
