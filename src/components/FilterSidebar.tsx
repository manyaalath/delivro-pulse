import { Filter, Upload } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { CalendarIcon } from "lucide-react";
import { format } from "date-fns";
import { cn } from "@/lib/utils";
import { Delivery, Filters } from "@/pages/Index";
import { CSVUpload } from "./CSVUpload";
import { useState } from "react";

interface FilterSidebarProps {
  filters: Filters;
  setFilters: (filters: Filters) => void;
  deliveries: Delivery[];
}

export const FilterSidebar = ({ filters, setFilters, deliveries }: FilterSidebarProps) => {
  const [showUpload, setShowUpload] = useState(false);

  const cities = [...new Set(deliveries.map((d) => d.city))];
  const regions = [...new Set(deliveries.map((d) => d.region_id))];
  const couriers = [...new Set(deliveries.map((d) => d.courier_id))];

  return (
    <>
      <aside className="w-80 border-r border-border bg-sidebar p-6 space-y-6 overflow-y-auto max-h-screen sticky top-16">
        <div className="flex items-center gap-2 mb-6">
          <Filter className="h-5 w-5 text-primary" />
          <h2 className="text-lg font-semibold">Filters</h2>
        </div>

        <div className="space-y-4">
          <div className="space-y-2">
            <Label>Date Range</Label>
            <Popover>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  className={cn(
                    "w-full justify-start text-left font-normal",
                    !filters.dateRange.from && "text-muted-foreground"
                  )}
                >
                  <CalendarIcon className="mr-2 h-4 w-4" />
                  {filters.dateRange.from ? (
                    filters.dateRange.to ? (
                      <>
                        {format(filters.dateRange.from, "LLL dd, y")} -{" "}
                        {format(filters.dateRange.to, "LLL dd, y")}
                      </>
                    ) : (
                      format(filters.dateRange.from, "LLL dd, y")
                    )
                  ) : (
                    <span>Pick a date range</span>
                  )}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0 pointer-events-auto bg-popover z-50" align="start">
                <Calendar
                  initialFocus
                  mode="range"
                  defaultMonth={filters.dateRange.from}
                  selected={{
                    from: filters.dateRange.from,
                    to: filters.dateRange.to,
                  }}
                  onSelect={(range) =>
                    setFilters({
                      ...filters,
                      dateRange: { from: range?.from, to: range?.to },
                    })
                  }
                  numberOfMonths={2}
                  className="pointer-events-auto"
                />
              </PopoverContent>
            </Popover>
          </div>

          <div className="space-y-2">
            <Label>City</Label>
            <Select
              value={filters.city}
              onValueChange={(value) =>
                setFilters({ ...filters, city: value === "all" ? "" : value })
              }
            >
              <SelectTrigger>
                <SelectValue placeholder="All Cities" />
              </SelectTrigger>
              <SelectContent className="bg-popover z-50">
                <SelectItem value="all">All Cities</SelectItem>
                {cities.map((city) => (
                  <SelectItem key={city} value={city}>
                    {city}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>Region</Label>
            <Select
              value={filters.region}
              onValueChange={(value) =>
                setFilters({ ...filters, region: value === "all" ? "" : value })
              }
            >
              <SelectTrigger>
                <SelectValue placeholder="All Regions" />
              </SelectTrigger>
              <SelectContent className="bg-popover z-50">
                <SelectItem value="all">All Regions</SelectItem>
                {regions.map((region) => (
                  <SelectItem key={region} value={region}>
                    {region}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>Courier</Label>
            <Select
              value={filters.courier}
              onValueChange={(value) =>
                setFilters({ ...filters, courier: value === "all" ? "" : value })
              }
            >
              <SelectTrigger>
                <SelectValue placeholder="All Couriers" />
              </SelectTrigger>
              <SelectContent className="bg-popover z-50">
                <SelectItem value="all">All Couriers</SelectItem>
                {couriers.map((courier) => (
                  <SelectItem key={courier} value={courier}>
                    {courier}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <Button
            variant="outline"
            className="w-full mt-4"
            onClick={() =>
              setFilters({
                dateRange: { from: undefined, to: undefined },
                city: "",
                region: "",
                courier: "",
              })
            }
          >
            Clear Filters
          </Button>

          <Button
            variant="default"
            className="w-full"
            onClick={() => setShowUpload(true)}
          >
            <Upload className="h-4 w-4 mr-2" />
            Upload CSV
          </Button>
        </div>
      </aside>

      <CSVUpload open={showUpload} onOpenChange={setShowUpload} />
    </>
  );
};
