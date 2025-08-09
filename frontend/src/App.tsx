import React, { useState, useEffect, useRef } from 'react';
import { Container, AppBar, Toolbar, Typography, Button, Grid, Paper } from '@mui/material';
import io from 'socket.io-client';
import { HandTracker } from './hand-tracker';

const socket = io('http://localhost:8000'); // Assuming backend is running on port 8000

function App() {
  const [prediction, setPrediction] = useState('');
  const [imagePrediction, setImagePrediction] = useState('');
  const [isRecognizing, setIsRecognizing] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const handTrackerRef = useRef<HandTracker | null>(null);

  useEffect(() => {
    socket.on('connect', () => {
      console.log('Connected to backend');
    });

    socket.on('prediction', (data) => {
      setPrediction(data.prediction);
    });

    handTrackerRef.current = new HandTracker();

    return () => {
      socket.off('connect');
      socket.off('prediction');
      handTrackerRef.current?.stop();
    };
  }, []);

  const startGestureRecognition = () => {
    if (videoRef.current) {
      handTrackerRef.current?.start(videoRef.current, (features, numHands) => {
        if (numHands === 1) {
          socket.emit('features', features);
        } else {
          socket.emit('features_dual', features);
        }
      });
      setIsRecognizing(true);
    }
  };

  const stopGestureRecognition = () => {
    handTrackerRef.current?.stop();
    setIsRecognizing(false);
  };

  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && handTrackerRef.current) {
      const result = await handTrackerRef.current.detectImage(file);
      if (result) {
        const { features, numHands } = result;
        const response = await fetch(`http://localhost:8000/predict_image?num_hands=${numHands}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ features }),
        });
        const data = await response.json();
        setImagePrediction(data.prediction);
      }
    }
  };

  return (
    <Container>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Hand Gesture Recognition
          </Typography>
        </Toolbar>
      </AppBar>

      <Grid container spacing={3} sx={{ mt: 3 }}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h5" gutterBottom>
              Real-time Recognition
            </Typography>
            <video ref={videoRef} width="100%" height="auto" autoPlay playsInline muted></video>
            <Button variant="contained" onClick={startGestureRecognition} sx={{ mr: 1 }} disabled={isRecognizing}>
              Start
            </Button>
            <Button variant="contained" onClick={stopGestureRecognition} color="secondary" disabled={!isRecognizing}>
              Stop
            </Button>
            <Typography variant="h6" sx={{ mt: 2 }}>
              Prediction: {prediction}
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h5" gutterBottom>
              Recognize from Image
            </Typography>
            <input type="file" accept="image/*" onChange={handleImageUpload} />
            <Typography variant="h6" sx={{ mt: 2 }}>
              Prediction: {imagePrediction}
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
}

export default App;