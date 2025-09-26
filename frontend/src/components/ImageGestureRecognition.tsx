import { useState, useRef, useCallback, useEffect } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { GestureResults } from './GestureResults';
import { HandLandmarks } from './HandLandmarks';
import { Alert, AlertDescription } from './ui/alert';
import { Upload, Image as ImageIcon, X } from 'lucide-react';
import { HandTracker } from '../hand-tracker';

// Define the structure for the gesture recognition result object.
interface GestureResult {
  handType: 'left' | 'right' | 'both' | 'none';
  gesture: string;
  confidence: number;
  landmarks?: any[]; // MediaPipe landmarks can be complex objects.
}

export function ImageGestureRecognition() {
  // State variables for managing the component.
  const [selectedImage, setSelectedImage] = useState<string>('');
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
  const [results, setResults] = useState<GestureResult>({
    handType: 'none',
    gesture: '',
    confidence: 0,
    landmarks: []
  });
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string>('');

  // Refs for the file input and HandTracker instance.
  const fileInputRef = useRef<HTMLInputElement>(null);
  const handTrackerRef = useRef<HandTracker | null>(null);

  // Initialize the HandTracker when the component mounts.
  useEffect(() => {
    handTrackerRef.current = new HandTracker();
  }, []);

  // Callback function to handle the image upload and prediction process.
  const handleImageUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Basic file validation.
    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file.');
      return;
    }
    if (file.size > 10 * 1024 * 1024) { // 10MB limit
      setError('Image file too large. Please select an image under 10MB.');
      return;
    }

    setError('');
    setIsProcessing(true);

    // Read the image file to display a preview.
    const reader = new FileReader();
    reader.onload = (e) => {
      const imageDataUrl = e.target?.result as string;
      setSelectedImage(imageDataUrl);

      // Create an Image object to get dimensions and process with HandTracker.
      const img = new Image();
      img.onload = async () => {
        // Scale image dimensions for display if necessary.
        const maxWidth = 640;
        const maxHeight = 480;
        let { width, height } = img;
        if (width > maxWidth || height > maxHeight) {
          const aspectRatio = width / height;
          if (width > height) {
            width = maxWidth;
            height = maxWidth / aspectRatio;
          } else {
            height = maxHeight;
            width = maxHeight * aspectRatio;
          }
        }
        setImageDimensions({ width, height });

        // Detect hand landmarks from the image file.
        if (handTrackerRef.current) {
          const result = await handTrackerRef.current.detectImage(file);
          if (result) {
            const { features, numHands, landmarks } = result;
            
            // Send the extracted features to the backend for prediction.
            const response = await fetch(`http://localhost:8000/predict_image?num_hands=${numHands}`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ features }),
            });
            const data = await response.json();
            
            // Update the results state with the prediction from the backend.
            setResults({ 
              handType: numHands === 1 ? 'right' : 'both', // Note: Handedness is not determined for images yet.
              gesture: data.prediction, 
              confidence: data.confidence, 
              landmarks 
            });
          }
        }
        setIsProcessing(false);
      };
      img.src = imageDataUrl;
    };
    reader.readAsDataURL(file);
  }, []);

  // Resets the state to allow for a new image upload.
  const clearImage = useCallback(() => {
    setSelectedImage('');
    setResults({ handType: 'none', gesture: '', confidence: 0, landmarks: [] });
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <ImageIcon className="w-5 h-5" />
            Image Hand Gesture Recognition
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {error && (
            <Alert>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <div className="flex gap-4">
            <Button onClick={() => fileInputRef.current?.click()} disabled={isProcessing} className="flex items-center gap-2">
              <Upload className="w-4 h-4" />
              {selectedImage ? 'Change Image' : 'Upload Image'}
            </Button>
            
            {selectedImage && (
              <Button onClick={clearImage} variant="outline" className="flex items-center gap-2">
                <X className="w-4 h-4" />
                Clear
              </Button>
            )}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            className="hidden"
          />

          <div className="flex gap-6">
            {/* Image Display Area */}
            <div className="relative">
              {selectedImage ? (
                <div className="relative">
                  <img
                    src={selectedImage}
                    alt="Uploaded"
                    className="rounded-lg border-2 border-gray-300 max-w-full"
                    style={{ 
                      width: `${imageDimensions.width}px`, 
                      height: `${imageDimensions.height}px`,
                      maxWidth: '640px',
                      maxHeight: '480px'
                    }}
                  />
                  
                  {results.landmarks && results.landmarks.length > 0 && !isProcessing && (
                    <HandLandmarks
                      landmarks={results.landmarks}
                      width={imageDimensions.width}
                      height={imageDimensions.height}
                      className="rounded-lg"
                    />
                  )}
                  
                  {isProcessing && (
                    <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center rounded-lg">
                      <div className="text-white text-center">
                        <div className="animate-spin w-8 h-8 border-4 border-white border-t-transparent rounded-full mx-auto mb-2"></div>
                        <p>Processing image...</p>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="w-96 h-64 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center bg-gray-50">
                  <div className="text-center text-gray-500">
                    <ImageIcon className="w-12 h-12 mx-auto mb-2 opacity-50" />
                    <p>No image selected</p>
                    <p className="text-sm">Upload an image to analyze</p>
                  </div>
                </div>
              )}
            </div>

            {/* Results Panel */}
            <GestureResults results={results} />
          </div>

          {selectedImage && (
            <div className="text-sm text-gray-600">
              <p><strong>Tip:</strong> For best results, ensure hands are clearly visible and well-lit in your image.</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}