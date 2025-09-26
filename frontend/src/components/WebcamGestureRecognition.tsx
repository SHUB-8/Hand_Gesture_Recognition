import { useEffect, useRef, useState, useCallback } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { GestureResults } from './GestureResults';
import { HandLandmarks } from './HandLandmarks';
import { Alert, AlertDescription } from './ui/alert';
import { Camera, CameraOff, RotateCcw } from 'lucide-react';
import io from 'socket.io-client';
import { HandTracker } from '../hand-tracker';

const socket = io('https://hand-gesture-recognition-8ulh.onrender.com');

interface GestureResult {
  handType: 'left' | 'right' | 'both' | 'none';
  gesture: string;
  confidence: number;
  landmarks?: number[][];
}

export function WebcamGestureRecognition() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState<string>('');
  const [results, setResults] = useState<GestureResult>({
    handType: 'none',
    gesture: '',
    confidence: 0,
    landmarks: []
  });
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const handTrackerRef = useRef<HandTracker | null>(null);

  const videoWidth = 640;
  const videoHeight = 480;

  useEffect(() => {
    const init = async () => {
      try {
        console.log('Initializing hand-tracker.ts');
        handTrackerRef.current = new HandTracker();
        await handTrackerRef.current.initialize();
        console.log('hand-tracker.ts initialized successfully');

    socket.on('prediction', (data: { prediction: string; confidence: number; hand?: 'left' | 'right'; hands?: ('left' | 'right')[] }) => {
      // If confidence is low, explicitly set the gesture to "Unknown".
      const finalGesture = data.confidence > 0.80 ? data.prediction : "Unknown";
      
      setResults(prevResults => ({
        ...prevResults,
        gesture: finalGesture,
        confidence: data.confidence,
        handType: data.hand || (data.hands ? 'both' : 'none'),
      }));
    });      } catch (error) {
        console.error('Error in useEffect:', error);
        setError('An unexpected error occurred. Please try again.');
      }
    };
    init();

    return () => {
      socket.off('prediction');
      handTrackerRef.current?.stop();
    };
  }, []);

  const startCamera = useCallback(async () => {
    try {
      setError('');
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: videoWidth, height: videoHeight }
      });
      
      setStream(mediaStream);
      
      if (videoRef.current && canvasRef.current) {
        videoRef.current.srcObject = mediaStream;
        videoRef.current.play();

        handTrackerRef.current?.start(videoRef.current, canvasRef.current, (features, numHands, landmarks, handedness) => {
          if (numHands === 1) {
            let hand = handedness[0][0].categoryName.toLowerCase();
            hand = hand === 'left' ? 'right' : 'left'; // Swap handedness
            socket.emit('features', { features, hand });
            setResults({ handType: hand, gesture: '', confidence: 0, landmarks });
          } else if (numHands === 2) {
            let hand1 = handedness[0][0].categoryName.toLowerCase();
            let hand2 = handedness[1][0].categoryName.toLowerCase();
            hand1 = hand1 === 'left' ? 'right' : 'left'; // Swap handedness
            hand2 = hand2 === 'left' ? 'right' : 'left'; // Swap handedness
            socket.emit('features_dual', { features, hands: [hand1, hand2] });
            setResults({ handType: 'both', gesture: '', confidence: 0, landmarks });
          } else {
            setResults({ handType: 'none', gesture: '', confidence: 0, landmarks: [] });
          }
        });
      }
      
      setIsActive(true);
    } catch (err) {
      setError('Failed to access camera. Please ensure camera permissions are granted.');
      console.error('Camera error:', err);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    handTrackerRef.current?.stop();
    setIsActive(false);
    setResults({ handType: 'none', gesture: '', confidence: 0, landmarks: [] });
  }, [stream]);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Camera className="w-5 h-5" />
            Real-time Hand Gesture Recognition
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {error && (
            <Alert>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          
          <div className="flex gap-4">
            <Button 
              onClick={startCamera} 
              disabled={isActive}
              className="flex items-center gap-2"
            >
              <Camera className="w-4 h-4" />
              Start Camera
            </Button>
            <Button 
              onClick={stopCamera} 
              disabled={!isActive}
              variant="outline"
              className="flex items-center gap-2"
            >
              <CameraOff className="w-4 h-4" />
              Stop Camera
            </Button>
            <Button 
              onClick={() => {
                stopCamera();
                setTimeout(startCamera, 100);
              }}
              disabled={!isActive}
              variant="outline"
              className="flex items-center gap-2"
            >
              <RotateCcw className="w-4 h-4" />
              Restart
            </Button>
          </div>

          <div className="flex gap-6">
            {/* Video Feed */}
            <div className="relative">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                width={videoWidth}
                height={videoHeight}
                className="rounded-lg bg-gray-900 border-2 border-gray-300"
                style={{ width: `${videoWidth}px`, height: `${videoHeight}px` }}
              />
              <canvas
                ref={canvasRef}
                width={videoWidth}
                height={videoHeight}
                className="absolute top-0 left-0 rounded-lg"
              />
              
              {/* Hand Landmarks Overlay */}
              {results.landmarks && results.landmarks.length > 0 && (
                <HandLandmarks
                  landmarks={results.landmarks}
                  width={videoWidth}
                  height={videoHeight}
                  className="rounded-lg"
                />
              )}
              
              {/* Status Overlay */}
              <div className="absolute top-2 left-2 bg-black bg-opacity-70 text-white px-2 py-1 rounded text-sm">
                {isActive ? '🔴 LIVE' : '⚫ STOPPED'}
              </div>
            </div>

            {/* Results Panel */}
            <GestureResults results={results} />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}