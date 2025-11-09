import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Navbar } from "@/components/Navbar";
import { FilterSidebar } from "@/components/FilterSidebar";
import { KPICards } from "@/components/KPICards";
import { MapView } from "@/components/MapView";
import { AnalyticsCharts } from "@/components/AnalyticsCharts";
import { TopCouriers } from "@/components/TopCouriers";
import { InsightsButton } from "@/components/InsightsButton";
import { ModelPerformance } from "@/components/ModelPerformance";
import { toast } from "sonner";

export interface Delivery {
  package_id: string;
  courier_id: string;
  city: string;
  region_id: string;
  lat: number;
  lng: number;
  accept_time: string;
  delivery_time: string;
  delay_min: number;
  status: string;
}

export interface Filters {
  dateRange: { from: Date | undefined; to: Date | undefined };
  city: string;
  region: string;
  courier: string;
}

const Index = () => {
  const [deliveries, setDeliveries] = useState<Delivery[]>([]);
  const [filteredDeliveries, setFilteredDeliveries] = useState<Delivery[]>([]);
  const [filters, setFilters] = useState<Filters>({
    dateRange: { from: undefined, to: undefined },
    city: "",
    region: "",
    courier: "",
  });
  const [loading, setLoading] = useState(true);

  // Fetch deliveries
  const fetchDeliveries = async () => {
    try {
      const { data, error } = await supabase
        .from("deliveries")
        .select("*")
        .order("delivery_time", { ascending: false });

      if (error) throw error;
      setDeliveries(data || []);
    } catch (error: any) {
      toast.error("Failed to fetch deliveries");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDeliveries();

    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchDeliveries, 30000);

    // Subscribe to realtime updates
    const channel = supabase
      .channel("deliveries-changes")
      .on(
        "postgres_changes",
        {
          event: "*",
          schema: "public",
          table: "deliveries",
        },
        () => {
          fetchDeliveries();
        }
      )
      .subscribe();

    return () => {
      clearInterval(interval);
      supabase.removeChannel(channel);
    };
  }, []);

  // Apply filters
  useEffect(() => {
    let filtered = [...deliveries];

    if (filters.dateRange.from) {
      filtered = filtered.filter(
        (d) => new Date(d.delivery_time) >= filters.dateRange.from!
      );
    }

    if (filters.dateRange.to) {
      filtered = filtered.filter(
        (d) => new Date(d.delivery_time) <= filters.dateRange.to!
      );
    }

    if (filters.city) {
      filtered = filtered.filter((d) => d.city === filters.city);
    }

    if (filters.region) {
      filtered = filtered.filter((d) => d.region_id === filters.region);
    }

    if (filters.courier) {
      filtered = filtered.filter((d) => d.courier_id === filters.courier);
    }

    setFilteredDeliveries(filtered);
  }, [deliveries, filters]);

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      
      <div className="flex">
        <FilterSidebar
          filters={filters}
          setFilters={setFilters}
          deliveries={deliveries}
        />

        <main className="flex-1 p-6 space-y-6">
          <KPICards deliveries={filteredDeliveries} loading={loading} />

          <ModelPerformance />

          <MapView />

          <AnalyticsCharts deliveries={filteredDeliveries} />

          <TopCouriers deliveries={filteredDeliveries} />
        </main>
      </div>

      <InsightsButton deliveries={filteredDeliveries} />
    </div>
  );
};

export default Index;
