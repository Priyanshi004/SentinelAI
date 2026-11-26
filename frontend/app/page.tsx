
"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  BarChart3,
  Clock3,
  LayoutDashboard,
  LoaderCircle,
  Network,
  RefreshCw,
  ScanSearch,
  Server,
  Shield,
  ShieldAlert,
  ShieldCheck,
  Wifi,
  WifiOff,
} from "lucide-react";

import {
  checkBackendHealth,
  getTrafficStatus,
  startTrafficSimulation,
  stopTrafficSimulation,
  type HealthResponse,
  type TrafficStatus,
} from "@/lib/api";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const COLORS = {
  green: "#36d6a5",
  red: "#f87171",
  blue: "#60a5fa",
  muted: "#8995a9",
};

function formatTime(value: string) {
  return new Date(value).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function StatCard({
  title,
  value,
  description,
  icon: Icon,
  color,
}: {
  title: string;
  value: string | number;
  description: string;
  icon: React.ElementType;
  color: string;
}) {
  return (
    <div className="rounded-xl border border-[#202c40] bg-[#101827] p-5">
      <div className="flex items-center justify-between">
        <p className="text-sm text-[#8995a9]">{title}</p>
        <div
          className="rounded-lg p-2"
          style={{ backgroundColor: `${color}18` }}
        >
          <Icon size={19} style={{ color }} />
        </div>
      </div>

      <h2 className="mt-4 text-3xl font-bold text-[#e8edf7]">
        {value}
      </h2>

      <p className="mt-2 text-xs text-[#8995a9]">{description}</p>
    </div>
  );
}

export default function DashboardPage() {
  const [traffic, setTraffic] = useState<TrafficStatus | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);
  const [error, setError] = useState("");

  const refreshDashboard = useCallback(async (manual = false) => {
    if (manual) setRefreshing(true);

    try {
      const [statusResult, healthResult] = await Promise.allSettled([
        getTrafficStatus(),
        checkBackendHealth(),
      ]);

      if (statusResult.status === "fulfilled") {
        setTraffic(statusResult.value);
        setError("");
      } else {
        setError("Unable to retrieve traffic statistics.");
      }

      if (healthResult.status === "fulfilled") {
        setHealth(healthResult.value);
      } else {
        setHealth(null);
      }
    } catch {
      setError("Unable to connect to the SentinelAI backend.");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void refreshDashboard();

    const interval = setInterval(() => {
      void refreshDashboard();
    }, 5000);

    return () => clearInterval(interval);
  }, [refreshDashboard]);

  const handleMonitoring = async () => {
    try {
      setActionLoading(true);
      setError("");

      if (traffic?.is_running) {
        await stopTrafficSimulation();
      } else {
        await startTrafficSimulation();
      }

      await refreshDashboard();
    } catch {
      setError("Monitoring action failed. Check the backend connection.");
    } finally {
      setActionLoading(false);
    }
  };

  const distributionData = useMemo(() => {
    if (!traffic) return [];

    return [
      { name: "Benign", value: traffic.benign_flows },
      { name: "Attacks", value: traffic.attack_flows },
    ];
  }, [traffic]);

  const activityData = useMemo(() => {
    if (!traffic) return [];

    return [...traffic.recent_events]
      .sort(
        (a, b) =>
          new Date(a.timestamp).getTime() -
          new Date(b.timestamp).getTime()
      )
      .slice(-10)
      .map((event) => ({
        flow: event.flow_id.replace("SIM-", "#"),
        attackProbability: Number(
          (event.attack_probability * 100).toFixed(1)
        ),
      }));
  }, [traffic]);

  const backendOnline = health?.status === "healthy";
  const modelOnline = health?.model_loaded === true;

  return (
    <main className="min-h-screen bg-[#080d17] text-[#e8edf7]">
      <div className="flex min-h-screen">

        {/* Sidebar */}
        <aside className="hidden w-64 shrink-0 border-r border-[#202c40] bg-[#0b1220] p-5 lg:flex lg:flex-col">
          <div className="flex items-center gap-3 px-2 py-4">
            <div className="rounded-xl bg-[#36d6a5]/10 p-2.5">
              <Shield className="text-[#36d6a5]" size={25} />
            </div>

            <div>
              <h1 className="text-xl font-bold tracking-wide">
                SENTINEL<span className="text-[#36d6a5]">AI</span>
              </h1>
              <p className="text-[10px] tracking-widest text-[#8995a9]">
                CYBERSECURITY PLATFORM
              </p>
            </div>
          </div>

          <div className="mt-9">
            <p className="mb-4 px-3 text-[10px] font-semibold tracking-[0.2em] text-[#66748b]">
              MAIN MENU
            </p>

            <nav className="space-y-2">
              <Link
                href="/"
                className="flex items-center gap-3 rounded-lg border border-[#36d6a5]/20 bg-[#36d6a5]/10 px-3 py-3 text-sm font-medium text-[#36d6a5]"
              >
                <LayoutDashboard size={19} />
                Overview
              </Link>

              <Link
                href="/threat-detection"
                className="flex items-center gap-3 rounded-lg px-3 py-3 text-sm text-[#8995a9] transition hover:bg-[#172235] hover:text-white"
              >
                <ScanSearch size={19} />
                Threat Detection
              </Link>

              <Link
                href="/network-traffic"
                className="flex items-center gap-3 rounded-lg px-3 py-3 text-sm text-[#8995a9] transition hover:bg-[#172235] hover:text-white"
              >
                <Network size={19} />
                Network Traffic
              </Link>
            </nav>
          </div>

          <div className="mt-auto rounded-xl border border-[#202c40] bg-[#101827] p-4">
            <div className="flex items-center gap-2">
              <ShieldCheck className="text-[#36d6a5]" size={20} />
              <span className="text-sm font-semibold">SentinelAI</span>
            </div>
            <p className="mt-2 text-xs leading-5 text-[#8995a9]">
              AI-powered intelligent cybersecurity threat detection and
              network monitoring.
            </p>
            <p className="mt-4 text-[10px] text-[#66748b]">
              MODEL: RANDOM FOREST
            </p>
          </div>
        </aside>

        {/* Main content */}
        <section className="min-w-0 flex-1">

          {/* Header */}
          <header className="sticky top-0 z-10 flex flex-wrap items-center justify-between gap-4 border-b border-[#202c40] bg-[#080d17]/95 px-5 py-4 backdrop-blur sm:px-8">
            <div>
              <h1 className="text-xl font-bold">Overview Dashboard</h1>
              <p className="mt-1 text-xs text-[#8995a9]">
                Monitor network activity and cybersecurity threats
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-2 rounded-lg border border-[#202c40] bg-[#101827] px-3 py-2">
                {backendOnline ? (
                  <Wifi size={15} className="text-[#36d6a5]" />
                ) : (
                  <WifiOff size={15} className="text-[#f87171]" />
                )}
                <span className="text-xs text-[#cbd5e1]">
                  {backendOnline ? "Backend Online" : "Backend Offline"}
                </span>
              </div>

              <button
                onClick={() => void refreshDashboard(true)}
                disabled={refreshing}
                className="flex items-center gap-2 rounded-lg border border-[#202c40] bg-[#101827] px-3 py-2 text-xs text-[#cbd5e1] transition hover:border-[#36d6a5]"
              >
                <RefreshCw
                  size={14}
                  className={refreshing ? "animate-spin" : ""}
                />
                Refresh
              </button>
            </div>
          </header>

          <div className="space-y-7 p-5 sm:p-8">

            {/* Welcome */}
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <p className="text-sm text-[#8995a9]">Security Overview</p>
                <h2 className="mt-1 text-2xl font-bold">
                  Welcome to SentinelAI
                </h2>
                <p className="mt-2 text-sm text-[#8995a9]">
                  Your network security monitoring center.
                </p>
              </div>

              <button
                onClick={() => void handleMonitoring()}
                disabled={actionLoading || !backendOnline}
                className={`flex items-center gap-2 rounded-lg px-5 py-3 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-50 ${traffic?.is_running
                    ? "border border-[#f87171]/30 bg-[#f87171]/10 text-[#f87171] hover:bg-[#f87171]/20"
                    : "bg-[#36d6a5] text-[#08111c] hover:bg-[#2bc394]"
                  }`}
              >
                {actionLoading ? (
                  <LoaderCircle size={17} className="animate-spin" />
                ) : traffic?.is_running ? (
                  <Activity size={17} />
                ) : (
                  <ShieldCheck size={17} />
                )}
                {traffic?.is_running
                  ? "Stop Monitoring"
                  : "Start Monitoring"}
              </button>
            </div>

            {/* Error */}
            {error && (
              <div className="flex items-center gap-3 rounded-lg border border-[#f87171]/30 bg-[#f87171]/10 p-4 text-sm text-[#fca5a5]">
                <AlertTriangle size={18} />
                {error}
              </div>
            )}

            {/* Monitoring status */}
            <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-[#202c40] bg-[#101827] p-4">
              <div className="flex items-center gap-3">
                <div
                  className={`h-2.5 w-2.5 rounded-full ${traffic?.is_running
                      ? "animate-pulse bg-[#36d6a5]"
                      : "bg-[#8995a9]"
                    }`}
                />
                <div>
                  <p className="text-sm font-semibold">
                    {traffic?.is_running
                      ? "Network monitoring is active"
                      : "Network monitoring is stopped"}
                  </p>
                  <p className="mt-1 text-xs text-[#8995a9]">
                    Dataset-driven traffic simulation
                  </p>
                </div>
              </div>

              <span className="rounded-md border border-[#202c40] px-3 py-1.5 text-xs text-[#8995a9]">
                CIC-IDS2017
              </span>
            </div>

            {/* Statistics */}
            <div>
              <div className="mb-4 flex items-center justify-between">
                <h3 className="font-semibold">Traffic Statistics</h3>
                <span className="text-xs text-[#66748b]">
                  Auto-refreshes every 5 seconds
                </span>
              </div>

              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                <StatCard
                  title="Total Flows"
                  value={loading ? "—" : (traffic?.total_flows ?? 0).toLocaleString()}
                  description="Total processed network flows"
                  icon={Activity}
                  color={COLORS.blue}
                />

                <StatCard
                  title="Detected Attacks"
                  value={loading ? "—" : (traffic?.attack_flows ?? 0).toLocaleString()}
                  description="Flows classified as attacks"
                  icon={ShieldAlert}
                  color={COLORS.red}
                />

                <StatCard
                  title="Benign Traffic"
                  value={loading ? "—" : (traffic?.benign_flows ?? 0).toLocaleString()}
                  description="Flows classified as benign"
                  icon={ShieldCheck}
                  color={COLORS.green}
                />

                <StatCard
                  title="Attack Rate"
                  value={`${(traffic?.attack_rate ?? 0).toFixed(2)}%`}
                  description="Percentage of detected attacks"
                  icon={AlertTriangle}
                  color="#fbbf24"
                />
              </div>
            </div>

            {/* Charts */}
            <div className="grid gap-5 xl:grid-cols-2">

              {/* Distribution */}
              <div className="rounded-xl border border-[#202c40] bg-[#101827] p-5">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-semibold">Traffic Distribution</h3>
                    <p className="mt-1 text-xs text-[#8995a9]">
                      Benign versus detected attack flows
                    </p>
                  </div>
                  <BarChart3 size={20} className="text-[#8995a9]" />
                </div>

                <div className="mt-5 h-64">
                  {loading ? (
                    <div className="flex h-full items-center justify-center">
                      <LoaderCircle className="animate-spin text-[#36d6a5]" />
                    </div>
                  ) : (traffic?.total_flows ?? 0) === 0 ? (
                    <div className="flex h-full flex-col items-center justify-center gap-2 text-[#8995a9]">
                      <Activity size={30} />
                      <p className="text-sm">No traffic processed yet</p>
                    </div>
                  ) : (
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={distributionData}
                          dataKey="value"
                          nameKey="name"
                          cx="50%"
                          cy="50%"
                          innerRadius={65}
                          outerRadius={95}
                          paddingAngle={4}
                        >
                          {distributionData.map((entry, index) => (
                            <Cell
                              key={entry.name}
                              fill={index === 0 ? COLORS.green : COLORS.red}
                            />
                          ))}
                        </Pie>
                        <Tooltip
                          contentStyle={{
                            background: "#101827",
                            border: "1px solid #29364b",
                            borderRadius: 8,
                            color: "#e8edf7",
                          }}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  )}
                </div>

                <div className="flex justify-center gap-6 text-xs">
                  <span className="flex items-center gap-2 text-[#8995a9]">
                    <span className="h-2.5 w-2.5 rounded-full bg-[#36d6a5]" />
                    Benign
                  </span>
                  <span className="flex items-center gap-2 text-[#8995a9]">
                    <span className="h-2.5 w-2.5 rounded-full bg-[#f87171]" />
                    Attacks
                  </span>
                </div>
              </div>

              {/* Attack probability */}
              <div className="rounded-xl border border-[#202c40] bg-[#101827] p-5">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-semibold">Recent Threat Activity</h3>
                    <p className="mt-1 text-xs text-[#8995a9]">
                      Attack probability for the latest 10 flows
                    </p>
                  </div>
                  <AlertTriangle size={20} className="text-[#f87171]" />
                </div>

                <div className="mt-5 h-64">
                  {activityData.length === 0 ? (
                    <div className="flex h-full flex-col items-center justify-center gap-2 text-[#8995a9]">
                      <BarChart3 size={30} />
                      <p className="text-sm">Waiting for traffic data</p>
                    </div>
                  ) : (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={activityData}>
                        <CartesianGrid
                          stroke="#29364b"
                          strokeDasharray="3 3"
                          vertical={false}
                        />
                        <XAxis
                          dataKey="flow"
                          stroke={COLORS.muted}
                          tick={{ fontSize: 10 }}
                        />
                        <YAxis
                          domain={[0, 100]}
                          stroke={COLORS.muted}
                          tick={{ fontSize: 10 }}
                          unit="%"
                        />
                        <Tooltip
                          contentStyle={{
                            background: "#101827",
                            border: "1px solid #29364b",
                            borderRadius: 8,
                            color: "#e8edf7",
                          }}
                          formatter={(value) => [`${value}%`, "Attack probability"]}
                        />
                        <Bar
                          dataKey="attackProbability"
                          fill={COLORS.red}
                          radius={[4, 4, 0, 0]}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </div>
              </div>
            </div>

            {/* Recent events */}
            <div className="overflow-hidden rounded-xl border border-[#202c40] bg-[#101827]">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#202c40] p-5">
                <div>
                  <h3 className="font-semibold">Recent Network Events</h3>
                  <p className="mt-1 text-xs text-[#8995a9]">
                    Latest processed traffic records
                  </p>
                </div>

                <Link
                  href="/network-traffic"
                  className="flex items-center gap-2 text-xs font-medium text-[#36d6a5] hover:underline"
                >
                  View all events <ArrowRight size={14} />
                </Link>
              </div>

              <div className="dashboard-scrollbar overflow-x-auto">
                <table className="w-full min-w-[700px] text-left text-sm">
                  <thead className="bg-[#0b1220] text-xs text-[#8995a9]">
                    <tr>
                      <th className="px-5 py-4 font-medium">Flow ID</th>
                      <th className="px-5 py-4 font-medium">Time</th>
                      <th className="px-5 py-4 font-medium">Destination Port</th>
                      <th className="px-5 py-4 font-medium">Packets</th>
                      <th className="px-5 py-4 font-medium">Prediction</th>
                      <th className="px-5 py-4 font-medium">Confidence</th>
                    </tr>
                  </thead>

                  <tbody className="divide-y divide-[#202c40]">
                    {loading ? (
                      <tr>
                        <td colSpan={6} className="px-5 py-10 text-center text-[#8995a9]">
                          Loading traffic events...
                        </td>
                      </tr>
                    ) : !traffic?.recent_events.length ? (
                      <tr>
                        <td colSpan={6} className="px-5 py-10 text-center text-[#8995a9]">
                          No traffic events yet. Start monitoring to see activity.
                        </td>
                      </tr>
                    ) : (
                      [...traffic.recent_events]
                        .sort(
                          (a, b) =>
                            new Date(b.timestamp).getTime() -
                            new Date(a.timestamp).getTime()
                        )
                        .slice(0, 8)
                        .map((event) => (
                          <tr
                            key={event.flow_id}
                            className="transition hover:bg-[#172235]"
                          >
                            <td className="px-5 py-4 font-mono text-xs text-[#cbd5e1]">
                              {event.flow_id}
                            </td>
                            <td className="px-5 py-4 text-xs text-[#8995a9]">
                              {formatTime(event.timestamp)}
                            </td>
                            <td className="px-5 py-4 text-xs">
                              {event.destination_port}
                            </td>
                            <td className="px-5 py-4 text-xs">
                              {event.total_packets}
                            </td>
                            <td className="px-5 py-4">
                              {event.prediction === "ATTACK" ? (
                                <span className="flex w-fit items-center gap-1.5 rounded-md bg-[#f87171]/10 px-2 py-1 text-xs text-[#f87171]">
                                  <ShieldAlert size={13} />
                                  ATTACK
                                </span>
                              ) : (
                                <span className="flex w-fit items-center gap-1.5 rounded-md bg-[#36d6a5]/10 px-2 py-1 text-xs text-[#36d6a5]">
                                  <ShieldCheck size={13} />
                                  BENIGN
                                </span>
                              )}
                            </td>
                            <td className="px-5 py-4 text-xs text-[#cbd5e1]">
                              {(event.confidence * 100).toFixed(1)}%
                            </td>
                          </tr>
                        ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* System information */}
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="flex items-center gap-4 rounded-xl border border-[#202c40] bg-[#101827] p-5">
                <div className="rounded-lg bg-[#36d6a5]/10 p-3">
                  <Server size={22} className="text-[#36d6a5]" />
                </div>
                <div>
                  <p className="text-sm font-semibold">Backend API</p>
                  <p className="mt-1 text-xs text-[#8995a9]">
                    {backendOnline ? "Connected" : "Not connected"}
                  </p>
                </div>
                <div className="ml-auto">
                  {backendOnline ? (
                    <ArrowUpRight className="text-[#36d6a5]" size={19} />
                  ) : (
                    <ArrowDownRight className="text-[#f87171]" size={19} />
                  )}
                </div>
              </div>

              <div className="flex items-center gap-4 rounded-xl border border-[#202c40] bg-[#101827] p-5">
                <div className="rounded-lg bg-[#60a5fa]/10 p-3">
                  <Shield size={22} className="text-[#60a5fa]" />
                </div>
                <div>
                  <p className="text-sm font-semibold">Random Forest Model</p>
                  <p className="mt-1 text-xs text-[#8995a9]">
                    {modelOnline
                      ? `Loaded · ${health?.features_expected ?? 70} features`
                      : "Not loaded"}
                  </p>
                </div>
                <div className="ml-auto">
                  {modelOnline ? (
                    <ShieldCheck className="text-[#36d6a5]" size={19} />
                  ) : (
                    <AlertTriangle className="text-[#f87171]" size={19} />
                  )}
                </div>
              </div>
            </div>

            {/* Footer */}
            <footer className="flex flex-wrap items-center justify-between gap-3 border-t border-[#202c40] pt-5 text-xs text-[#66748b]">
              <p>SentinelAI · AI-Powered Cybersecurity Monitoring</p>
              <p>Data source: CIC-IDS2017 simulated traffic</p>
            </footer>
          </div>
        </section>
      </div>
    </main>
  );
}
