"use client"

import {
  BarChart as RechartsBarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart as RechartsRadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
} from 'recharts';

interface BarChartProps {
  data: Array<{
    name: string;
    value: number;
    target?: number;
  }>;
  xAxis: string;
  yAxis: string;
}

interface RadarChartProps {
  data: Array<{
    subject: string;
    A: number;
    B: number;
    fullMark: number;
  }>;
}

export function BarChart({ data, xAxis, yAxis }: BarChartProps) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <RechartsBarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={xAxis} />
        <YAxis />
        <Tooltip />
        <Bar dataKey="value" fill="#3b82f6" />
        {data[0]?.target && <Bar dataKey="target" fill="#93c5fd" />}
      </RechartsBarChart>
    </ResponsiveContainer>
  );
}

export function RadarChart({ data }: RadarChartProps) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <RechartsRadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
        <PolarGrid />
        <PolarAngleAxis dataKey="subject" />
        <PolarRadiusAxis angle={30} domain={[0, 100]} />
        <Radar
          name="Current"
          dataKey="A"
          stroke="#3b82f6"
          fill="#3b82f6"
          fillOpacity={0.6}
        />
        <Radar
          name="Target"
          dataKey="B"
          stroke="#93c5fd"
          fill="#93c5fd"
          fillOpacity={0.6}
        />
        <Legend />
      </RechartsRadarChart>
    </ResponsiveContainer>
  );
} 