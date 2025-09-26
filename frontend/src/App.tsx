import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import { WebcamGestureRecognition } from "./components/WebcamGestureRecognition";
import { ImageGestureRecognition } from "./components/ImageGestureRecognition";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./components/ui/card";
import { Hand, Camera, Image } from "lucide-react";

export default function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-4">
          <div className="flex items-center justify-center gap-3">
            <Hand className="w-8 h-8 text-indigo-600" />
            <h1 className="text-4xl font-bold text-gray-900">
              Hand Gesture Recognition
            </h1>
          </div>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Advanced hand gesture recognition powered by MediaPipe. Analyze gestures in real-time 
            through your webcam or upload images for detailed hand landmark analysis.
          </p>
        </div>



        {/* Main Application */}
        <Tabs defaultValue="webcam" className="space-y-6">
          <TabsList className="grid w-full grid-cols-2 max-w-md mx-auto">
            <TabsTrigger value="webcam" className="flex items-center gap-2">
              <Camera className="w-4 h-4" />
              Real-time
            </TabsTrigger>
            <TabsTrigger value="image" className="flex items-center gap-2">
              <Image className="w-4 h-4" />
              Image Upload
            </TabsTrigger>
          </TabsList>

          <TabsContent value="webcam">
            <WebcamGestureRecognition />
          </TabsContent>

          <TabsContent value="image">
            <ImageGestureRecognition />
          </TabsContent>
        </Tabs>

        {/* Footer */}
        <div className="text-center text-sm text-gray-500 border-t pt-6">
          <p>
            This demo simulates MediaPipe hand detection. In a production environment, 
            integrate with actual MediaPipe libraries for real gesture recognition.
          </p>
        </div>
      </div>
    </div>
  );
}