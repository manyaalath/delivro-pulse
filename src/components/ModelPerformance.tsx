import { useState, useEffect, forwardRef, useImperativeHandle } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Loader2 } from "lucide-react";

interface ModelMetrics {
  accuracy: number;
  f1_score: number;
  precision: number;
  recall: number;
}

interface InsightsResponse {
  model_metrics: ModelMetrics;
}

export const ModelPerformance = forwardRef<{ refreshMetrics: () => void }>((props, ref) => {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch("http://127.0.0.1:8001/insights");
      if (!response.ok) {
        throw new Error("Failed to fetch model metrics");
      }
      const data: InsightsResponse = await response.json();
      setMetrics(data.model_metrics);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  useImperativeHandle(ref, () => ({
    refreshMetrics: fetchMetrics,
  }));

  useEffect(() => {
    fetchMetrics();
  }, []);

  if (loading) {
    return (
      <Card className="p-6">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Loader2 className="h-5 w-5 animate-spin" />
            Model Performance Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">Loading metrics...</p>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="p-6">
        <CardHeader>
          <CardTitle>Model Performance Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-red-500">Error: {error}</p>
        </CardContent>
      </Card>
    );
  }

  if (!metrics) {
    return (
      <Card className="p-6">
        <CardHeader>
          <CardTitle>Model Performance Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">No metrics available</p>
        </CardContent>
      </Card>
    );
  }

  const formatMetric = (value: number) => (value * 100).toFixed(2) + "%";

  const getMetricColor = (value: number) => {
    if (value >= 0.8) return "bg-green-500";
    if (value >= 0.6) return "bg-yellow-500";
    return "bg-red-500";
  };

  return (
    <Card className="p-6">
      <CardHeader>
        <CardTitle>Model Performance Metrics</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <Badge className={`${getMetricColor(metrics.accuracy)} text-white mb-2`}>
              Accuracy
            </Badge>
            <p className="text-2xl font-bold">{formatMetric(metrics.accuracy)}</p>
          </div>
          <div className="text-center">
            <Badge className={`${getMetricColor(metrics.f1_score)} text-white mb-2`}>
              F1 Score
            </Badge>
            <p className="text-2xl font-bold">{formatMetric(metrics.f1_score)}</p>
          </div>
          <div className="text-center">
            <Badge className={`${getMetricColor(metrics.precision)} text-white mb-2`}>
              Precision
            </Badge>
            <p className="text-2xl font-bold">{formatMetric(metrics.precision)}</p>
          </div>
          <div className="text-center">
            <Badge className={`${getMetricColor(metrics.recall)} text-white mb-2`}>
              Recall
            </Badge>
            <p className="text-2xl font-bold">{formatMetric(metrics.recall)}</p>
          </div>
        </div>
        <p className="text-sm text-muted-foreground mt-4">
          These metrics evaluate the model's ability to predict late deliveries.
        </p>
      </CardContent>
    </Card>
  );
});
