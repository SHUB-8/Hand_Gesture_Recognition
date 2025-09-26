import { useEffect, useRef } from 'react';
import { DrawingUtils, HandLandmarker } from '@mediapipe/tasks-vision';

interface HandLandmarksProps {
  landmarks: any[];
  width: number;
  height: number;
  className?: string;
}

export function HandLandmarks({ landmarks, width, height, className = "" }: HandLandmarksProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !landmarks.length) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const drawingUtils = new DrawingUtils(ctx);

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    for (const landmark of landmarks) {
        drawingUtils.drawConnectors(landmark, HandLandmarker.HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 5 });
        drawingUtils.drawLandmarks(landmark, { color: '#FF0000', lineWidth: 2 });
    }

  }, [landmarks, width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className={`absolute top-0 left-0 pointer-events-none ${className}`}
      style={{ width: `${width}px`, height: `${height}px` }}
    />
  );
}