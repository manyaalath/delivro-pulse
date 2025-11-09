import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { deliveries } = await req.json();
    const LOVABLE_API_KEY = Deno.env.get('LOVABLE_API_KEY');

    if (!LOVABLE_API_KEY) {
      throw new Error('LOVABLE_API_KEY is not configured');
    }

    // Prepare analytics summary
    const totalDeliveries = deliveries.length;
    const lateDeliveries = deliveries.filter((d: any) => d.delay_min > 0);
    const cityStats: Record<string, any> = {};
    const courierStats: Record<string, any> = {};
    const hourlyStats: Record<number, any> = {};

    deliveries.forEach((d: any) => {
      // City stats
      if (!cityStats[d.city]) {
        cityStats[d.city] = { total: 0, late: 0, totalDelay: 0 };
      }
      cityStats[d.city].total++;
      if (d.delay_min > 0) {
        cityStats[d.city].late++;
        cityStats[d.city].totalDelay += d.delay_min;
      }

      // Courier stats
      if (!courierStats[d.courier_id]) {
        courierStats[d.courier_id] = { total: 0, onTime: 0 };
      }
      courierStats[d.courier_id].total++;
      if (d.delay_min <= 0) {
        courierStats[d.courier_id].onTime++;
      }

      // Hourly stats
      const hour = new Date(d.delivery_time).getHours();
      if (!hourlyStats[hour]) {
        hourlyStats[hour] = { total: 0, late: 0, totalDelay: 0 };
      }
      hourlyStats[hour].total++;
      if (d.delay_min > 0) {
        hourlyStats[hour].late++;
        hourlyStats[hour].totalDelay += d.delay_min;
      }
    });

    // Find insights
    const worstCity = Object.entries(cityStats)
      .map(([city, stats]: [string, any]) => ({
        city,
        avgDelay: stats.late > 0 ? stats.totalDelay / stats.late : 0,
      }))
      .sort((a, b) => b.avgDelay - a.avgDelay)[0];

    const bestCourier = Object.entries(courierStats)
      .map(([courier, stats]: [string, any]) => ({
        courier,
        onTimeRate: (stats.onTime / stats.total) * 100,
        total: stats.total,
      }))
      .filter(c => c.total >= 5)
      .sort((a, b) => b.onTimeRate - a.onTimeRate)[0];

    const peakHour = Object.entries(hourlyStats)
      .map(([hour, stats]: [string, any]) => ({
        hour: parseInt(hour),
        avgDelay: stats.late > 0 ? stats.totalDelay / stats.late : 0,
      }))
      .sort((a, b) => b.avgDelay - a.avgDelay)[0];

    const summary = `Dataset contains ${totalDeliveries} deliveries with ${lateDeliveries.length} late deliveries (${Math.round((lateDeliveries.length / totalDeliveries) * 100)}%).

Key statistics:
- Worst performing city: ${worstCity?.city || 'N/A'}
- Average delay in worst city: ${Math.round(worstCity?.avgDelay || 0)} minutes
- Best courier: ${bestCourier?.courier || 'N/A'} with ${Math.round(bestCourier?.onTimeRate || 0)}% on-time rate
- Peak delay hour: ${peakHour?.hour || 'N/A'}:00 with ${Math.round(peakHour?.avgDelay || 0)} min average delay`;

    const prompt = `You are a logistics analytics expert. Analyze this delivery data and provide actionable insights in 3-4 concise bullet points.

${summary}

Focus on:
1. Main performance patterns
2. Operational bottlenecks
3. Specific recommendations for improvement

Keep it clear and actionable.`;

    const response = await fetch('https://ai.gateway.lovable.dev/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${LOVABLE_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'google/gemini-2.5-flash',
        messages: [
          { role: 'system', content: 'You are a logistics analytics expert providing concise, actionable insights.' },
          { role: 'user', content: prompt }
        ],
      }),
    });

    if (!response.ok) {
      if (response.status === 429) {
        return new Response(
          JSON.stringify({ error: 'Rate limit exceeded. Please try again later.' }),
          { status: 429, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }
      if (response.status === 402) {
        return new Response(
          JSON.stringify({ error: 'Payment required. Please add credits to your workspace.' }),
          { status: 402, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }
      const errorText = await response.text();
      console.error('AI Gateway error:', response.status, errorText);
      throw new Error('Failed to generate insights');
    }

    const data = await response.json();
    const insights = data.choices[0].message.content;

    return new Response(
      JSON.stringify({ insights }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );

  } catch (error: any) {
    console.error('Error in generate-insights:', error);
    return new Response(
      JSON.stringify({ error: error?.message || 'Failed to generate insights' }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});
