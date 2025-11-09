import { Card } from "@/components/ui/card";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { Delivery } from "@/pages/Index";
import { format } from "date-fns";

interface AnalyticsChartsProps {
  deliveries: Delivery[];
}

export const AnalyticsCharts = ({ deliveries }: AnalyticsChartsProps) => {
  // Late deliveries by city
  const cityData = deliveries.reduce((acc, delivery) => {
    const city = delivery.city;
    if (!acc[city]) {
      acc[city] = { city, lateCount: 0 };
    }
    if (delivery.delay_min > 0) {
      acc[city].lateCount++;
    }
    return acc;
  }, {} as Record<string, { city: string; lateCount: number }>);

  const cityChartData = Object.values(cityData);

  // Average delay over time (by day)
  const timeData = deliveries.reduce((acc, delivery) => {
    const date = format(new Date(delivery.delivery_time), "MMM dd");
    if (!acc[date]) {
      acc[date] = { date, totalDelay: 0, count: 0 };
    }
    acc[date].totalDelay += Math.max(0, delivery.delay_min);
    acc[date].count++;
    return acc;
  }, {} as Record<string, { date: string; totalDelay: number; count: number }>);

  const timeChartData = Object.values(timeData)
    .map((item) => ({
      date: item.date,
      avgDelay: Math.round(item.totalDelay / item.count),
    }))
    .sort((a, b) => a.date.localeCompare(b.date));

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Late Deliveries by City</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={cityChartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis
              dataKey="city"
              stroke="#888"
              style={{ fontSize: "12px" }}
            />
            <YAxis stroke="#888" style={{ fontSize: "12px" }} />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1a1a1a",
                border: "1px solid #333",
                borderRadius: "8px",
              }}
            />
            <Bar dataKey="lateCount" fill="#FF5555" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Average Delay Over Time</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={timeChartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis
              dataKey="date"
              stroke="#888"
              style={{ fontSize: "12px" }}
            />
            <YAxis stroke="#888" style={{ fontSize: "12px" }} />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1a1a1a",
                border: "1px solid #333",
                borderRadius: "8px",
              }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="avgDelay"
              stroke="#00FFFF"
              strokeWidth={2}
              dot={{ fill: "#00FFFF", r: 4 }}
              name="Avg Delay (min)"
            />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  );
};
