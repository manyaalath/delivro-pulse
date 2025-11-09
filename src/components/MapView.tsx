import { useEffect, useState } from "react";
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { Card } from "@/components/ui/card";
import { toast } from "sonner";

// Fix for default markers in Leaflet
import L from 'leaflet';
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

interface GeoPoint {
  lat: number;
  lng: number;
  delay_min: number;
  late_flag: number;
  city: string;
  region_id: string;
}

export const MapView = () => {
  const [geoPoints, setGeoPoints] = useState<GeoPoint[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchGeoData = async () => {
      console.log("🌍 Fetching /geo data...");
      try {
        const response = await fetch("http://localhost:8000/geo");
        if (!response.ok) {
          throw new Error("Failed to fetch geo data");
        }
        const data = await response.json();
        setGeoPoints(data.geo_data);
        console.log(`✅ Geo data loaded: ${data.geo_data.length} points.`);
      } catch (error) {
        console.error("❌ Failed to load map data:", error);
        toast.error("Failed to load map data");
      } finally {
        setLoading(false);
      }
    };

    fetchGeoData();
  }, []);

  if (loading) {
    return (
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Delivery Locations</h3>
        <div className="w-full h-[500px] rounded-2xl shadow-md border border-gray-700 flex items-center justify-center bg-gray-100">
          <p className="text-muted-foreground">🧭 Loading geospatial data...</p>
        </div>
      </Card>
    );
  }

  return (
    <Card className="p-6">
      <h3 className="text-lg font-semibold mb-4">Delivery Locations</h3>
      <div className="w-full h-[500px] rounded-2xl shadow-md border border-gray-700">
        <MapContainer
          center={[20.5937, 78.9629]}
          zoom={4}
          style={{ height: '100%', width: '100%' }}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {geoPoints.map((point, index) => {
            const avgDelay = point.delay_min;
            const isHighDelay = avgDelay > 15;
            const isLowDelay = avgDelay < 10;

            return (
              <Marker
                key={index}
                position={[point.lat, point.lng]}
                icon={new L.Icon({
                  iconUrl: isHighDelay
                    ? 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png'
                    : isLowDelay
                    ? 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-green.png'
                    : 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-blue.png',
                  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
                  iconSize: [25, 41],
                  iconAnchor: [12, 41],
                  popupAnchor: [1, -34],
                  shadowSize: [41, 41]
                })}
              >
                <Popup>
                  <div style={{ padding: '8px', background: '#1a1a1a', color: 'white', borderRadius: '8px' }}>
                    <p style={{ margin: 0, fontWeight: 600, color: isHighDelay ? '#FF5555' : isLowDelay ? '#00FF88' : '#5555FF' }}>
                      City: {point.city}
                    </p>
                    <p style={{ margin: '4px 0 0', fontSize: '12px', color: '#888' }}>
                      Region: {point.region_id}<br/>
                      Avg Delay: {avgDelay.toFixed(2)} min
                    </p>
                  </div>
                </Popup>
              </Marker>
            );
          })}
        </MapContainer>
      </div>
      <div className="flex items-center gap-6 mt-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-green-500" />
          <span>Low Delay {'<'}10 min</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-blue-500" />
          <span>Medium Delay</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-red-500" />
          <span>High Delay {'>'}15 min</span>
        </div>
      </div>
    </Card>
  );
};
