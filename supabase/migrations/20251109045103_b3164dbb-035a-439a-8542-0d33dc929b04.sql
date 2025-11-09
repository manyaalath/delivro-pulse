-- Create deliveries table for logistics analytics
CREATE TABLE IF NOT EXISTS public.deliveries (
  package_id TEXT PRIMARY KEY,
  courier_id TEXT NOT NULL,
  city TEXT NOT NULL,
  region_id TEXT NOT NULL,
  lat FLOAT8 NOT NULL,
  lng FLOAT8 NOT NULL,
  accept_time TIMESTAMPTZ NOT NULL,
  delivery_time TIMESTAMPTZ NOT NULL,
  delay_min INT8 GENERATED ALWAYS AS (
    EXTRACT(EPOCH FROM (delivery_time - accept_time)) / 60
  ) STORED,
  status TEXT DEFAULT 'delivered',
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE public.deliveries ENABLE ROW LEVEL SECURITY;

-- Create policy for public read access (analytics dashboard)
CREATE POLICY "Allow public read access" 
ON public.deliveries 
FOR SELECT 
USING (true);

-- Create policy for public insert (for CSV uploads)
CREATE POLICY "Allow public insert" 
ON public.deliveries 
FOR INSERT 
WITH CHECK (true);

-- Create indexes for better query performance
CREATE INDEX idx_deliveries_city ON public.deliveries(city);
CREATE INDEX idx_deliveries_region ON public.deliveries(region_id);
CREATE INDEX idx_deliveries_courier ON public.deliveries(courier_id);
CREATE INDEX idx_deliveries_delivery_time ON public.deliveries(delivery_time);
CREATE INDEX idx_deliveries_delay ON public.deliveries(delay_min);

-- Insert sample data for demo
INSERT INTO public.deliveries (package_id, courier_id, city, region_id, lat, lng, accept_time, delivery_time, status) VALUES
  ('PKG001', 'C101', 'Delhi', 'DL01', 28.7041, 77.1025, NOW() - INTERVAL '2 hours', NOW() - INTERVAL '1 hour 45 minutes', 'delivered'),
  ('PKG002', 'C102', 'Mumbai', 'MH01', 19.0760, 72.8777, NOW() - INTERVAL '3 hours', NOW() - INTERVAL '2 hours 30 minutes', 'delivered'),
  ('PKG003', 'C101', 'Delhi', 'DL01', 28.5355, 77.3910, NOW() - INTERVAL '4 hours', NOW() - INTERVAL '3 hours', 'delivered'),
  ('PKG004', 'C103', 'Bangalore', 'KA01', 12.9716, 77.5946, NOW() - INTERVAL '5 hours', NOW() - INTERVAL '4 hours 40 minutes', 'delivered'),
  ('PKG005', 'C102', 'Mumbai', 'MH01', 19.2183, 72.9781, NOW() - INTERVAL '6 hours', NOW() - INTERVAL '5 hours 50 minutes', 'delivered'),
  ('PKG006', 'C104', 'Chennai', 'TN01', 13.0827, 80.2707, NOW() - INTERVAL '1 hour', NOW() - INTERVAL '30 minutes', 'delivered'),
  ('PKG007', 'C101', 'Delhi', 'DL02', 28.6139, 77.2090, NOW() - INTERVAL '2 hours', NOW() - INTERVAL '1 hour 20 minutes', 'delivered'),
  ('PKG008', 'C105', 'Kolkata', 'WB01', 22.5726, 88.3639, NOW() - INTERVAL '3 hours', NOW() - INTERVAL '2 hours 10 minutes', 'delivered')
ON CONFLICT (package_id) DO NOTHING;