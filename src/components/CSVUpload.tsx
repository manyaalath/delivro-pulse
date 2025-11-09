import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";
import { Upload } from "lucide-react";

interface CSVUploadProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export const CSVUpload = ({ open, onOpenChange }: CSVUploadProps) => {
  const [uploading, setUploading] = useState(false);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    try {
      const text = await file.text();
      const lines = text.split("\n").filter((line) => line.trim());
      const headers = lines[0].split(",").map((h) => h.trim());

      const records = lines.slice(1).map((line) => {
        const values = line.split(",").map((v) => v.trim());
        const record: any = {};
        headers.forEach((header, index) => {
          record[header] = values[index];
        });
        return record;
      });

      const { error } = await supabase.from("deliveries").insert(records);

      if (error) throw error;

      toast.success(`Successfully uploaded ${records.length} deliveries`);
      onOpenChange(false);
    } catch (error: any) {
      console.error("Upload error:", error);
      toast.error("Failed to upload CSV: " + error.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-card">
        <DialogHeader>
          <DialogTitle>Upload CSV Data</DialogTitle>
          <DialogDescription>
            Upload a CSV file with delivery data. The file should include columns: package_id, courier_id, city, region_id, lat, lng, accept_time, delivery_time
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="border-2 border-dashed border-border rounded-lg p-8 text-center">
            <Upload className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <Input
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              disabled={uploading}
              className="cursor-pointer"
            />
          </div>

          <div className="text-sm text-muted-foreground">
            <p className="font-semibold mb-2">CSV Format Example:</p>
            <code className="block bg-secondary p-3 rounded text-xs">
              package_id,courier_id,city,region_id,lat,lng,accept_time,delivery_time
              <br />
              PKG001,C101,Delhi,DL01,28.7041,77.1025,2024-01-01 10:00:00,2024-01-01 11:00:00
            </code>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};
