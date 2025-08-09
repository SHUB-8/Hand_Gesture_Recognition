import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

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
  private runningMode: 'IMAGE' | 'VIDEO' = 'VIDEO';
  private videoStream: MediaStream | undefined;
  private lastVideoTime = -1;
  private animationFrameId: number | undefined;

  constructor() {
    this.initializeHandLandmarker();
  }

  private async initializeHandLandmarker() {
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm'
    );
    this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
        delegate: 'GPU',
      },
      runningMode: this.runningMode,
      numHands: 2,
    });
  }

  async start(videoElement: HTMLVideoElement, callback: (features: number[], numHands: number) => void) {
    if (!this.handLandmarker) {
      await this.initializeHandLandmarker();
    }

    if (this.runningMode !== 'VIDEO') {
        this.runningMode = 'VIDEO';
        await this.handLandmarker.setOptions({ runningMode: 'VIDEO' });
    }

    this.videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoElement.srcObject = this.videoStream;
    videoElement.addEventListener('loadeddata', () => {
      this.predictWebcam(videoElement, callback);
    });
  }

  stop() {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }
    if (this.videoStream) {
      this.videoStream.getTracks().forEach(track => track.stop());
    }
  }

  private predictWebcam = (videoElement: HTMLVideoElement, callback: (features: number[], numHands: number) => void) => {
    if (!this.handLandmarker) return;

    const nowInMs = Date.now();
    if (videoElement.currentTime !== this.lastVideoTime) {
      this.lastVideoTime = videoElement.currentTime;
      const handLandmarkerResult = this.handLandmarker.detectForVideo(videoElement, nowInMs);

      if (handLandmarkerResult.landmarks && handLandmarkerResult.landmarks.length > 0) {
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
    }

    this.animationFrameId = requestAnimationFrame(() => this.predictWebcam(videoElement, callback));
  };

  async detectImage(file: File): Promise<{features: number[], numHands: number} | undefined> {
    if (!this.handLandmarker) {
      await this.initializeHandLandmarker();
    }

    if (this.runningMode !== 'IMAGE') {
        this.runningMode = 'IMAGE';
        await this.handLandmarker.setOptions({ runningMode: 'IMAGE' });
    }

    const image = new Image();
    image.src = URL.createObjectURL(file);
    return new Promise((resolve) => {
      image.onload = () => {
        const handLandmarkerResult = this.handLandmarker.detect(image);
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
