import { HandLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';

// Feature extraction functions
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

function getUpperTriangle(matrix: number[][]) {
  const upperTriangle: number[] = [];
  for (let i = 0; i < matrix.length; i++) {
    for (let j = i + 1; j < matrix[i].length; j++) {
      upperTriangle.push(matrix[i][j]);
    }
  }
  return upperTriangle;
}

function extractFeatures(landmarks: any[]) {
  const distMatrix = calculateDistanceMatrix(landmarks);
  return getUpperTriangle(distMatrix);
}

export class HandTracker {
  private handLandmarker: HandLandmarker | undefined;
  private currentRunningMode: 'IMAGE' | 'VIDEO' = 'VIDEO'; // Track current mode
  private videoStream: MediaStream | undefined;
  private lastVideoTime = -1;
  private animationFrameId: number | undefined;
  private drawingUtils: DrawingUtils | undefined;
  private canvasCtx: CanvasRenderingContext2D | null = null;
  private frameCounter = 0;
  private frameSkip = 3; // Process every 3rd frame

  constructor() {
    // No initial handLandmarker creation here, it will be created on demand
  }

  private async createHandLandmarker(runningMode: 'IMAGE' | 'VIDEO') {
    if (this.handLandmarker) {
      this.handLandmarker.close(); // Close existing instance if any
    }
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm'
    );
    this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
        delegate: 'GPU',
      },
      runningMode: runningMode,
      numHands: 2,
    });
    this.currentRunningMode = runningMode;
  }

  async start(videoElement: HTMLVideoElement, canvasElement: HTMLCanvasElement, callback: (features: number[], numHands: number) => void) {
    if (this.currentRunningMode !== 'VIDEO' || !this.handLandmarker) {
      await this.createHandLandmarker('VIDEO');
    }

    this.videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoElement.srcObject = this.videoStream;
    await videoElement.play(); // Ensure video playback

    this.canvasCtx = canvasElement.getContext('2d');
    this.drawingUtils = new DrawingUtils(this.canvasCtx!); // Initialize drawing utilities

    videoElement.addEventListener('loadeddata', () => {
      this.predictWebcam(videoElement, canvasElement, callback);
    });
  }

  stop() {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }
    if (this.videoStream) {
      this.videoStream.getTracks().forEach(track => track.stop());
    }
    if (this.handLandmarker) {
      this.handLandmarker.close(); // Close the hand landmarker when stopping
      this.handLandmarker = undefined;
    }
  }

  private predictWebcam = (videoElement: HTMLVideoElement, canvasElement: HTMLCanvasElement, callback: (features: number[], numHands: number) => void) => {
    if (!this.handLandmarker || !this.canvasCtx || !this.drawingUtils) return;

    const nowInMs = Date.now();
    if (videoElement.currentTime !== this.lastVideoTime) {
      this.lastVideoTime = videoElement.currentTime;

      this.frameCounter++;
      if (this.frameCounter % this.frameSkip === 0) {
        const handLandmarkerResult = this.handLandmarker.detectForVideo(videoElement, nowInMs);

        // Clear canvas and draw video frame
        this.canvasCtx.save();
        this.canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        this.canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

        if (handLandmarkerResult.landmarks && handLandmarkerResult.landmarks.length > 0) {
          for (const landmarks of handLandmarkerResult.landmarks) {
            this.drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 2.5 }); // Halved
            this.drawingUtils.drawLandmarks(landmarks, { color: '#FF0000', lineWidth: 0.25 }); // Halved
          }

          if (handLandmarkerResult.landmarks.length === 1) {
            const features = extractFeatures(handLandmarkerResult.landmarks[0]);
            callback(features, 1);
          } else {
            const features1 = extractFeatures(handLandmarkerResult.landmarks[0]);
            const features2 = extractFeatures(handLandmarkerResult.landmarks[1]);
            const combinedFeatures = [...features1, ...features2];
            callback(combinedFeatures, 2);
          }
        }
        this.canvasCtx.restore();
      }
    }

    this.animationFrameId = requestAnimationFrame(() => this.predictWebcam(videoElement, canvasElement, callback));
  };

  async detectImage(file: File): Promise<{features: number[], numHands: number} | undefined> {
    if (this.currentRunningMode !== 'IMAGE' || !this.handLandmarker) {
      await this.createHandLandmarker('IMAGE');
    }

    const image = new Image();
    image.src = URL.createObjectURL(file);
    return new Promise((resolve) => {
      image.onload = () => {
        const handLandmarkerResult = this.handLandmarker!.detect(image);
        if (handLandmarkerResult.landmarks && handLandmarkerResult.landmarks.length > 0) {
          if (handLandmarkerResult.landmarks.length === 1) {
            const features = extractFeatures(handLandmarkerResult.landmarks[0]);
            resolve({ features, numHands: 1 });
          } else {
            const features1 = extractFeatures(handLandmarkerResult.landmarks[0]);
            const features2 = extractFeatures(handLandmarkerResult.landmarks[1]);
            const combinedFeatures = [...features1, ...features2];
            resolve({ features: combinedFeatures, numHands: 2 });
          }
        } else {
          resolve(undefined);
        }
      };
    });
  }
}
