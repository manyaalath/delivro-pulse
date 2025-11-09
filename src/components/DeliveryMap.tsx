import { useEffect, useRef, useState } from "react";
import mapboxgl from "mapbox-gl";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Delivery } from "@/pages/Index";
import { toast } from "sonner";

interface DeliveryMapProps {
  deliveries: Delivery[];
}

export const DeliveryMap = ({ deliveries }: DeliveryMapProps) => {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const [mapboxToken, setMapboxToken] = useState("");
  const [tokenSet, setTokenSet] = useState(false);
  const markersRef = useRef<mapboxgl.Marker[]>([]);

  useEffect(() => {
    if (!mapContainer.current || !tokenSet || !mapboxToken) return;

    try {
      mapboxgl.accessToken = mapboxToken;

      map.current = new mapboxgl.Map({
        container: mapContainer.current,
        style: "mapbox://styles/mapbox/dark-v11",
        center: [78.9629, 20.5937], // India center
        zoom: 4,
      });

      map.current.addControl(new mapboxgl.NavigationControl(), "top-right");
    } catch (error) {
      console.error("Map initialization error:", error);
      toast.error("Failed to initialize map. Please check your Mapbox token.");
    }

    return () => {
      markersRef.current.forEach((marker) => marker.remove());
      markersRef.current = [];
      map.current?.remove();
    };
  }, [tokenSet, mapboxToken]);

  useEffect(() => {
    if (!map.current || !tokenSet) return;

    // Clear existing markers
    markersRef.current.forEach((marker) => marker.remove());
    markersRef.current = [];

    // Add new markers
    deliveries.forEach((delivery) => {
      const isLate = delivery.delay_min > 0;
      
      const el = document.createElement("div");
      el.className = "marker";
      el.style.width = "12px";
      el.style.height = "12px";
      el.style.borderRadius = "50%";
      el.style.backgroundColor = isLate ? "#FF5555" : "#00FF88";
      el.style.border = "2px solid white";
      el.style.cursor = "pointer";
      el.style.boxShadow = isLate
        ? "0 0 10px rgba(255, 85, 85, 0.8)"
        : "0 0 10px rgba(0, 255, 136, 0.8)";

      const popup = new mapboxgl.Popup({ offset: 25 }).setHTML(`
        <div style="padding: 8px; background: #1a1a1a; color: white; border-radius: 8px;">
          <p style="margin: 0; font-weight: 600; color: ${isLate ? '#FF5555' : '#00FF88'}">
            ${delivery.package_id}
          </p>
          <p style="margin: 4px 0 0; font-size: 12px; color: #888">
            Courier: ${delivery.courier_id}<br/>
            City: ${delivery.city}<br/>
            ${isLate ? `Delay: ${delivery.delay_min} min` : 'On Time'}
          </p>
        </div>
      `);

      const marker = new mapboxgl.Marker(el)
        .setLngLat([delivery.lng, delivery.lat])
        .setPopup(popup)
        .addTo(map.current!);

      markersRef.current.push(marker);
    });
  }, [deliveries, tokenSet]);

  if (!tokenSet) {
    return (
      <Card className="p-6">
        <div className="max-w-md space-y-4">
          <div>
            <h3 className="text-lg font-semibold mb-2">Mapbox Configuration</h3>
            <p className="text-sm text-muted-foreground mb-4">
              Enter your Mapbox public token to enable the interactive delivery map.
              Get your token at{" "}
              <a
                href="https://mapbox.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:underline"
              >
                mapbox.com
              </a>
            </p>
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="mapbox-token">Mapbox Public Token</Label>
            <Input
              id="mapbox-token"
              type="text"
              placeholder="pk.eyJ1..."
              value={mapboxToken}
              onChange={(e) => setMapboxToken(e.target.value)}
            />
          </div>

          <button
            onClick={() => {
              if (mapboxToken.trim()) {
                setTokenSet(true);
                toast.success("Map initialized successfully!");
              } else {
                toast.error("Please enter a valid Mapbox token");
              }
            }}
            className="w-full px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
          >
            Initialize Map
          </button>
        </div>
      </Card>
    );
  }

  return (
    <Card className="p-6">
      <h3 className="text-lg font-semibold mb-4">Delivery Locations</h3>
      <div
        ref={mapContainer}
        className="h-[500px] w-full rounded-lg border border-border overflow-hidden"
      />
      <div className="flex items-center gap-6 mt-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-success shadow-glow-green" />
          <span>On-Time</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-accent shadow-glow-red" />
          <span>Late</span>
        </div>
      </div>
    </Card>
  );
};
