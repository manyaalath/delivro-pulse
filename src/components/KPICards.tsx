import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Package, AlertCircle, Clock, CheckCircle } from "lucide-react";
import { Delivery } from "@/pages/Index";

interface KPICardsProps {
  deliveries: Delivery[];
  loading: boolean;
}

export const KPICards = ({ deliveries, loading }: KPICardsProps) => {
  const totalDeliveries = deliveries.length;
  const lateDeliveries = deliveries.filter((d) => d.delay_min > 0).length;
  const avgDelay =
    deliveries.length > 0
      ? Math.round(
          deliveries.reduce((acc, d) => acc + Math.max(0, d.delay_min), 0) /
            deliveries.length
        )
      : 0;
  const onTimePercentage =
    totalDeliveries > 0
      ? Math.round(((totalDeliveries - lateDeliveries) / totalDeliveries) * 100)
      : 100;

  const kpis = [
    {
      title: "Total Deliveries",
      value: totalDeliveries,
      icon: Package,
      color: "text-primary",
      bg: "bg-primary/10",
    },
    {
      title: "Late Deliveries",
      value: lateDeliveries,
      icon: AlertCircle,
      color: "text-accent",
      bg: "bg-accent/10",
    },
    {
      title: "Average Delay",
      value: `${avgDelay} min`,
      icon: Clock,
      color: "text-yellow-500",
      bg: "bg-yellow-500/10",
    },
    {
      title: "On-Time %",
      value: `${onTimePercentage}%`,
      icon: CheckCircle,
      color: "text-success",
      bg: "bg-success/10",
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {kpis.map((kpi) => (
        <Card
          key={kpi.title}
          className="p-6 border-border hover:border-primary/50 transition-all duration-300 hover:shadow-glow-cyan"
        >
          {loading ? (
            <div className="space-y-3">
              <Skeleton className="h-4 w-24" />
              <Skeleton className="h-8 w-16" />
            </div>
          ) : (
            <>
              <div className="flex items-center justify-between mb-4">
                <p className="text-sm text-muted-foreground">{kpi.title}</p>
                <div className={`p-2 rounded-lg ${kpi.bg}`}>
                  <kpi.icon className={`h-5 w-5 ${kpi.color}`} />
                </div>
              </div>
              <p className={`text-3xl font-bold ${kpi.color}`}>{kpi.value}</p>
            </>
          )}
        </Card>
      ))}
    </div>
  );
};
