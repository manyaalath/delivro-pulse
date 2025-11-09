import { useEffect, useState } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
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
  city: string;
  region_id: string;
  lat: number;
  lng: number;
  avg_delay: number;
}

function FitToBounds({ points }: { points: GeoPoint[] }) {
  const map = useMap();
  useEffect(() => {
    if (points.length > 0) {
      const bounds = L.latLngBounds(points.map(p => [p.lat, p.lng]));
      map.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [points, map]);
  return null;
}

export const MapView = () => {
  const [geoPoints, setGeoPoints] = useState<GeoPoint[]>([]);

  useEffect(() => {
    console.log("🌍 Fetching /geo data…");
    fetch("http://127.0.0.1:8001/geo")
      .then((res) => res.json())
      .then((data) => {
        setGeoPoints(data.points);
        console.log(`✅ Geo data loaded: ${data.points.length} points.`);
      })
      .catch((err) => console.error("❌ Failed to fetch geo data:", err));
  }, []);

  return (
    <div className="w-full h-[500px] rounded-2xl shadow-lg border border-gray-700 mt-4">
      {geoPoints.length === 0 && <p className="text-center text-gray-400">🧭 Loading geospatial data…</p>}
      <MapContainer
        center={[20.5937, 78.9629]}
        zoom={4}
        style={{ height: '100%', width: '100%' }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <FitToBounds points={geoPoints} />
        {geoPoints.map((point, index) => (
          <Marker
            key={index}
            position={[point.lat, point.lng]}
            icon={L.icon({
              iconUrl:
                point.avg_delay > 15
                  ? "https://maps.gstatic.com/mapfiles/ms2/micons/red-dot.png"
                  : point.avg_delay > 10
                  ? "https://maps.gstatic.com/mapfiles/ms2/micons/orange-dot.png"
                  : "https://maps.gstatic.com/mapfiles/ms2/micons/green-dot.png",
              iconSize: [30, 30],
              iconAnchor: [15, 30],
            })}
          >
            <Popup>
              <b>{point.city}</b><br />
              Region: {point.region_id}<br />
              Avg Delay: {point.avg_delay} min
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
};
