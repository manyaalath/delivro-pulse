import { useState } from "react";
import { Brain } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { toast } from "sonner";
import { Delivery } from "@/pages/Index";

interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
}

interface InsightsButtonProps {
  deliveries: Delivery[];
  onInsightsRefresh?: () => void; // Optional callback to refresh other components
}

export const InsightsButton = ({ deliveries, onInsightsRefresh }: InsightsButtonProps) => {
  const [open, setOpen] = useState(false);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [loading, setLoading] = useState(false);

  const generateInsights = async () => {
    setLoading(true);
    try {
      // Optional retrain
      await fetch("http://localhost:8000/train", {
        method: "POST",
      });

      // Get insights
      const response = await fetch("http://localhost:8000/insights");
      if (!response.ok) {
        throw new Error("Failed to fetch insights");
      }
      const data = await response.json();
      setMetrics(data.model_metrics);
      setOpen(true);

      // Log success
      console.log("🧠 Insights refreshed");

      // Trigger refresh of other components if callback provided
      if (onInsightsRefresh) {
        onInsightsRefresh();
      }
    } catch (error: any) {
      console.error("Insights error:", error);
      toast.error("Failed to generate insights: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Button
        onClick={generateInsights}
        disabled={loading || deliveries.length === 0}
        className="fixed bottom-8 right-8 rounded-full h-14 px-6 shadow-glow-cyan bg-primary text-primary-foreground hover:bg-primary/90"
      >
        <Brain className="h-5 w-5 mr-2" />
        {loading ? "Analyzing..." : "🧠 Generate Insights"}
      </Button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="bg-card max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-primary" />
              Model Performance Metrics
            </DialogTitle>
            <DialogDescription>
              Latest model metrics after retraining
            </DialogDescription>
          </DialogHeader>

          {metrics && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Accuracy</p>
                <p className="text-2xl font-bold">{(metrics.accuracy * 100).toFixed(2)}%</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Precision</p>
                <p className="text-2xl font-bold">{(metrics.precision * 100).toFixed(2)}%</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Recall</p>
                <p className="text-2xl font-bold">{(metrics.recall * 100).toFixed(2)}%</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">F1 Score</p>
                <p className="text-2xl font-bold">{(metrics.f1 * 100).toFixed(2)}%</p>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
};
