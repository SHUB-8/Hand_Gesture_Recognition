import { Badge } from "./ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Progress } from "./ui/progress";

interface GestureResult {
  handType: 'left' | 'right' | 'both' | 'none';
  gesture: string;
  confidence: number;
  landmarks?: number[][];
}

interface GestureResultsProps {
  results: GestureResult;
}

export function GestureResults({ results }: GestureResultsProps) {
  const getHandTypeColor = (handType: string) => {
    switch (handType) {
      case 'left': return 'bg-blue-100 text-blue-800';
      case 'right': return 'bg-green-100 text-green-800';
      case 'both': return 'bg-purple-100 text-purple-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-green-500';
    if (confidence >= 0.6) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <Card className="w-full max-w-md">
      <CardHeader>
        <CardTitle>Detection Results</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Hand Type */}
        <div className="flex items-center justify-between">
          <span className="text-sm">Hand Type:</span>
          <Badge className={getHandTypeColor(results.handType)}>
            {results.handType === 'none' ? 'No hands detected' : results.handType.toUpperCase()}
          </Badge>
        </div>

        {/* Predicted Gesture */}
        <div className="flex items-center justify-between">
          <span className="text-sm">Gesture:</span>
          <Badge variant="outline">
            {results.gesture || 'Unknown'}
          </Badge>
        </div>

        {/* Confidence Score */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm">Confidence:</span>
            <span className="text-sm">{(results.confidence * 100).toFixed(1)}%</span>
          </div>
          <Progress 
            value={results.confidence * 100} 
            className="h-2"
          />
        </div>

        {/* Landmarks Count */}
        {results.landmarks && (
          <div className="flex items-center justify-between">
            <span className="text-sm">Landmarks:</span>
            <Badge variant="secondary">
              {results.landmarks.length} points
            </Badge>
          </div>
        )}
      </CardContent>
    </Card>
  );
}