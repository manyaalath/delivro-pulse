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
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";
import { Delivery } from "@/pages/Index";

interface InsightsButtonProps {
  deliveries: Delivery[];
}

export const InsightsButton = ({ deliveries }: InsightsButtonProps) => {
  const [open, setOpen] = useState(false);
  const [insights, setInsights] = useState("");
  const [loading, setLoading] = useState(false);

  const generateInsights = async () => {
    setLoading(true);
    try {
      const { data, error } = await supabase.functions.invoke("generate-insights", {
        body: { deliveries },
      });

      if (error) throw error;

      setInsights(data.insights);
      setOpen(true);
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
              AI-Generated Insights
            </DialogTitle>
            <DialogDescription>
              Analysis of your logistics data
            </DialogDescription>
          </DialogHeader>

          <div className="prose prose-invert max-w-none">
            <p className="whitespace-pre-wrap">{insights}</p>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
};
