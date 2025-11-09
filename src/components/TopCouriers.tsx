import { Card } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Delivery } from "@/pages/Index";

interface TopCouriersProps {
  deliveries: Delivery[];
}

export const TopCouriers = ({ deliveries }: TopCouriersProps) => {
  const courierStats = deliveries.reduce((acc, delivery) => {
    const courierId = delivery.courier_id;
    if (!acc[courierId]) {
      acc[courierId] = {
        courier_id: courierId,
        total: 0,
        onTime: 0,
      };
    }
    acc[courierId].total++;
    if (delivery.delay_min <= 0) {
      acc[courierId].onTime++;
    }
    return acc;
  }, {} as Record<string, { courier_id: string; total: number; onTime: number }>);

  const topCouriers = Object.values(courierStats)
    .filter((c) => c.total >= 20)
    .map((c) => ({
      ...c,
      onTimeRate: Math.round((c.onTime / c.total) * 100),
    }))
    .sort((a, b) => b.onTimeRate - a.onTimeRate)
    .slice(0, 10);

  return (
    <Card className="p-6">
      <h3 className="text-lg font-semibold mb-4">
        Top 10 Couriers by On-Time Rate
        <span className="text-sm text-muted-foreground ml-2">
          (minimum 20 deliveries)
        </span>
      </h3>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Rank</TableHead>
            <TableHead>Courier ID</TableHead>
            <TableHead>Total Deliveries</TableHead>
            <TableHead>On-Time Deliveries</TableHead>
            <TableHead>On-Time Rate</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {topCouriers.length === 0 ? (
            <TableRow>
              <TableCell colSpan={5} className="text-center text-muted-foreground">
                No couriers with 20+ deliveries found
              </TableCell>
            </TableRow>
          ) : (
            topCouriers.map((courier, index) => (
              <TableRow key={courier.courier_id}>
                <TableCell className="font-medium">
                  {index + 1}
                </TableCell>
                <TableCell>{courier.courier_id}</TableCell>
                <TableCell>{courier.total}</TableCell>
                <TableCell>{courier.onTime}</TableCell>
                <TableCell>
                  <Badge
                    variant={courier.onTimeRate >= 90 ? "default" : "secondary"}
                    className={
                      courier.onTimeRate >= 90
                        ? "bg-success text-success-foreground"
                        : ""
                    }
                  >
                    {courier.onTimeRate}%
                  </Badge>
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </Card>
  );
};
