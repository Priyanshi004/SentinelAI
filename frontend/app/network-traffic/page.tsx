
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
    Activity,
    AlertTriangle,
    ArrowDownUp,
    CheckCircle2,
    Clock,
    Database,
    LoaderCircle,
    Network,
    Play,
    RefreshCw,
    ShieldCheck,
    ShieldAlert,
    Square,
    Wifi,
} from "lucide-react";

import {
    getNextTrafficFlow,
    getTrafficStatus,
    startTrafficSimulation,
    stopTrafficSimulation,
    type TrafficFlow,
    type TrafficStatus,
} from "@/lib/api";

export default function NetworkTrafficPage() {
    const [traffic, setTraffic] = useState<TrafficStatus | null>(null);
    const [loading, setLoading] = useState(true);
    const [actionLoading, setActionLoading] = useState(false);
    const [error, setError] = useState("");

    const processingRef = useRef(false);

    const isRunning = traffic?.is_running ?? false;

    const loadStatus = useCallback(async () => {
        try {
            const data = await getTrafficStatus();
            setTraffic(data);
            setError("");
        } catch {
            setError("Unable to connect to the traffic monitoring API.");
        } finally {
            setLoading(false);
        }
    }, []);

    const processTraffic = useCallback(async () => {
        if (processingRef.current) return;

        processingRef.current = true;

        try {
            await getNextTrafficFlow();

            const updatedStatus = await getTrafficStatus();
            setTraffic(updatedStatus);
            setError("");
        } catch {
            setError("Unable to process the next traffic flow.");
        } finally {
            processingRef.current = false;
        }
    }, []);

    useEffect(() => {
        loadStatus();
    }, [loadStatus]);

    useEffect(() => {
        if (!isRunning) return;

        const interval = setInterval(() => {
            processTraffic();
        }, 2500);

        return () => clearInterval(interval);
    }, [isRunning, processTraffic]);

    const handleStart = async () => {
        setActionLoading(true);
        setError("");

        try {
            const data = await startTrafficSimulation();
            setTraffic(data);
        } catch {
            setError("Could not start traffic simulation. Check your backend.");
        } finally {
            setActionLoading(false);
        }
    };

    const handleStop = async () => {
        setActionLoading(true);
        setError("");

        try {
            const data = await stopTrafficSimulation();
            setTraffic(data);
        } catch {
            setError("Could not stop traffic simulation.");
        } finally {
            setActionLoading(false);
        }
    };

    const handleRefresh = async () => {
        setLoading(true);
        await loadStatus();
    };

    const formatTime = (timestamp: string) => {
        return new Date(timestamp).toLocaleTimeString("en-IN", {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
        });
    };

    const formatNumber = (value: number) => {
        return new Intl.NumberFormat("en-IN").format(value);
    };

    const StatCard = ({
        title,
        value,
        subtitle,
        icon: Icon,
        iconColor,
        iconBg,
    }: {
        title: string;
        value: string | number;
        subtitle: string;
        icon: React.ElementType;
        iconColor: string;
        iconBg: string;
    }) => (
        <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5">
            <div className="flex items-center justify-between">
                <p className="text-sm font-medium text-slate-400">{title}</p>
                <div className={`rounded-xl p-3 ${iconBg}`}>
                    <Icon size={21} className={iconColor} />
                </div>
            </div>

            <h3 className="mt-5 text-3xl font-bold tracking-tight text-white">
                {value}
            </h3>

            <p className="mt-2 text-xs text-slate-500">{subtitle}</p>
        </div>
    );

    const events: TrafficFlow[] = traffic?.recent_events ?? [];

    return (
        <main className="min-h-screen bg-[#080d19] p-5 text-white md:p-8">
            <div className="mx-auto max-w-7xl space-y-7">

                {/* Page Header */}
                <div className="flex flex-col justify-between gap-5 md:flex-row md:items-center">
                    <div>
                        <div className="mb-3 flex items-center gap-2 text-sm text-cyan-400">
                            <Network size={17} />
                            <span>SECURITY OPERATIONS CENTER</span>
                        </div>

                        <h1 className="text-3xl font-bold tracking-tight md:text-4xl">
                            Network Traffic Monitoring
                        </h1>

                        <p className="mt-2 text-sm text-slate-400">
                            Monitor network flows and detect suspicious traffic
                            using the SentinelAI Random Forest model.
                        </p>
                    </div>

                    <button
                        onClick={handleRefresh}
                        disabled={loading}
                        className="flex items-center justify-center gap-2 rounded-xl border border-slate-700 bg-slate-900 px-4 py-3 text-sm font-medium text-slate-300 transition hover:bg-slate-800 disabled:opacity-50"
                    >
                        <RefreshCw
                            size={16}
                            className={loading ? "animate-spin" : ""}
                        />
                        Refresh
                    </button>
                </div>

                {/* Simulation Status */}
                <section className="flex flex-col justify-between gap-5 rounded-2xl border border-slate-800 bg-slate-900/80 p-5 md:flex-row md:items-center">
                    <div className="flex items-center gap-4">
                        <div
                            className={`rounded-xl p-3 ${isRunning
                                    ? "bg-emerald-500/10"
                                    : "bg-slate-800"
                                }`}
                        >
                            <Activity
                                size={25}
                                className={
                                    isRunning
                                        ? "text-emerald-400"
                                        : "text-slate-400"
                                }
                            />
                        </div>

                        <div>
                            <h2 className="font-semibold text-white">
                                Traffic Simulation
                            </h2>

                            <div className="mt-1 flex items-center gap-2">
                                <span
                                    className={`h-2 w-2 rounded-full ${isRunning
                                            ? "animate-pulse bg-emerald-400"
                                            : "bg-slate-500"
                                        }`}
                                />

                                <span
                                    className={`text-sm ${isRunning
                                            ? "text-emerald-400"
                                            : "text-slate-400"
                                        }`}
                                >
                                    {isRunning ? "Monitoring Active" : "Stopped"}
                                </span>
                            </div>
                        </div>
                    </div>

                    <div className="flex flex-wrap gap-3">
                        <button
                            onClick={handleStart}
                            disabled={isRunning || actionLoading}
                            className="flex items-center gap-2 rounded-xl bg-emerald-500 px-5 py-3 text-sm font-semibold text-white transition hover:bg-emerald-600 disabled:cursor-not-allowed disabled:opacity-40"
                        >
                            {actionLoading ? (
                                <LoaderCircle size={17} className="animate-spin" />
                            ) : (
                                <Play size={17} />
                            )}
                            Start Monitoring
                        </button>

                        <button
                            onClick={handleStop}
                            disabled={!isRunning || actionLoading}
                            className="flex items-center gap-2 rounded-xl bg-rose-500/10 px-5 py-3 text-sm font-semibold text-rose-400 transition hover:bg-rose-500/20 disabled:cursor-not-allowed disabled:opacity-40"
                        >
                            <Square size={15} />
                            Stop Monitoring
                        </button>
                    </div>
                </section>

                {/* Error Message */}
                {error && (
                    <div className="flex items-center gap-3 rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-300">
                        <AlertTriangle size={19} />
                        {error}
                    </div>
                )}

                {/* Statistics */}
                <section>
                    <div className="mb-4 flex items-center gap-2">
                        <Activity size={19} className="text-cyan-400" />
                        <h2 className="text-lg font-semibold">
                            Traffic Overview
                        </h2>
                    </div>

                    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
                        <StatCard
                            title="Total Flows"
                            value={formatNumber(traffic?.total_flows ?? 0)}
                            subtitle="Processed network flows"
                            icon={ArrowDownUp}
                            iconColor="text-cyan-400"
                            iconBg="bg-cyan-400/10"
                        />

                        <StatCard
                            title="Benign Traffic"
                            value={formatNumber(traffic?.benign_flows ?? 0)}
                            subtitle="Flows classified as BENIGN"
                            icon={ShieldCheck}
                            iconColor="text-emerald-400"
                            iconBg="bg-emerald-400/10"
                        />

                        <StatCard
                            title="Detected Attacks"
                            value={formatNumber(traffic?.attack_flows ?? 0)}
                            subtitle="Flows classified as ATTACK"
                            icon={ShieldAlert}
                            iconColor="text-rose-400"
                            iconBg="bg-rose-400/10"
                        />

                        <StatCard
                            title="Attack Rate"
                            value={`${(traffic?.attack_rate ?? 0).toFixed(2)}%`}
                            subtitle="Percentage of processed flows"
                            icon={AlertTriangle}
                            iconColor="text-amber-400"
                            iconBg="bg-amber-400/10"
                        />
                    </div>
                </section>

                {/* Traffic Distribution */}
                <section className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5">
                    <div className="mb-5 flex items-center justify-between">
                        <div>
                            <h2 className="font-semibold text-white">
                                Traffic Distribution
                            </h2>
                            <p className="mt-1 text-xs text-slate-500">
                                Classification breakdown of processed flows
                            </p>
                        </div>

                        <Database size={20} className="text-cyan-400" />
                    </div>

                    {(() => {
                        const total = traffic?.total_flows ?? 0;
                        const benign = traffic?.benign_flows ?? 0;
                        const attacks = traffic?.attack_flows ?? 0;

                        const benignPercent =
                            total > 0 ? (benign / total) * 100 : 0;

                        const attackPercent =
                            total > 0 ? (attacks / total) * 100 : 0;

                        return (
                            <>
                                <div className="mb-3 flex h-4 overflow-hidden rounded-full bg-slate-800">
                                    <div
                                        className="bg-emerald-400 transition-all duration-500"
                                        style={{ width: `${benignPercent}%` }}
                                    />
                                    <div
                                        className="bg-rose-500 transition-all duration-500"
                                        style={{ width: `${attackPercent}%` }}
                                    />
                                </div>

                                <div className="flex flex-wrap gap-6 text-sm">
                                    <div className="flex items-center gap-2">
                                        <span className="h-3 w-3 rounded-sm bg-emerald-400" />
                                        <span className="text-slate-400">
                                            Benign: {benignPercent.toFixed(2)}%
                                        </span>
                                    </div>

                                    <div className="flex items-center gap-2">
                                        <span className="h-3 w-3 rounded-sm bg-rose-500" />
                                        <span className="text-slate-400">
                                            Attacks: {attackPercent.toFixed(2)}%
                                        </span>
                                    </div>
                                </div>
                            </>
                        );
                    })()}
                </section>

                {/* Recent Events */}
                <section className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/80">
                    <div className="flex flex-col justify-between gap-3 border-b border-slate-800 p-5 sm:flex-row sm:items-center">
                        <div>
                            <h2 className="font-semibold text-white">
                                Recent Traffic Events
                            </h2>
                            <p className="mt-1 text-xs text-slate-500">
                                Latest processed network flows
                            </p>
                        </div>

                        <div className="flex items-center gap-2 text-xs text-slate-400">
                            <Wifi size={15} className="text-cyan-400" />
                            {events.length} recent events
                        </div>
                    </div>

                    <div className="overflow-x-auto">
                        <table className="w-full min-w-[850px] text-left text-sm">
                            <thead className="bg-slate-800/50 text-xs uppercase text-slate-400">
                                <tr>
                                    <th className="px-5 py-4">Flow ID</th>
                                    <th className="px-5 py-4">Timestamp</th>
                                    <th className="px-5 py-4">Destination Port</th>
                                    <th className="px-5 py-4">Packets</th>
                                    <th className="px-5 py-4">Prediction</th>
                                    <th className="px-5 py-4">Attack Probability</th>
                                    <th className="px-5 py-4">Confidence</th>
                                </tr>
                            </thead>

                            <tbody className="divide-y divide-slate-800">
                                {events.length > 0 ? (
                                    events.map((event) => (
                                        <tr
                                            key={event.flow_id}
                                            className="transition hover:bg-slate-800/40"
                                        >
                                            <td className="px-5 py-4 font-mono text-xs text-cyan-400">
                                                {event.flow_id}
                                            </td>

                                            <td className="px-5 py-4 text-slate-400">
                                                <span className="flex items-center gap-2">
                                                    <Clock size={14} />
                                                    {formatTime(event.timestamp)}
                                                </span>
                                            </td>

                                            <td className="px-5 py-4 text-slate-300">
                                                {event.destination_port}
                                            </td>

                                            <td className="px-5 py-4 text-slate-300">
                                                {formatNumber(event.total_packets)}
                                            </td>

                                            <td className="px-5 py-4">
                                                <span
                                                    className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold ${event.prediction === "ATTACK"
                                                            ? "bg-rose-500/10 text-rose-400"
                                                            : "bg-emerald-500/10 text-emerald-400"
                                                        }`}
                                                >
                                                    {event.prediction === "ATTACK" ? (
                                                        <ShieldAlert size={13} />
                                                    ) : (
                                                        <CheckCircle2 size={13} />
                                                    )}
                                                    {event.prediction}
                                                </span>
                                            </td>

                                            <td className="px-5 py-4 font-medium text-slate-300">
                                                {(event.attack_probability * 100).toFixed(2)}%
                                            </td>

                                            <td className="px-5 py-4 text-slate-300">
                                                {(event.confidence * 100).toFixed(2)}%
                                            </td>
                                        </tr>
                                    ))
                                ) : (
                                    <tr>
                                        <td
                                            colSpan={7}
                                            className="px-5 py-12 text-center"
                                        >
                                            <Database
                                                size={30}
                                                className="mx-auto mb-3 text-slate-600"
                                            />
                                            <p className="text-sm text-slate-400">
                                                No traffic events available.
                                            </p>
                                            <p className="mt-1 text-xs text-slate-500">
                                                Start monitoring to process network flows.
                                            </p>
                                        </td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </section>

                {/* Data Source Information */}
                <section className="flex flex-col gap-3 rounded-2xl border border-cyan-500/20 bg-cyan-500/5 p-5 sm:flex-row sm:items-center">
                    <div className="rounded-xl bg-cyan-500/10 p-3">
                        <Database size={22} className="text-cyan-400" />
                    </div>

                    <div>
                        <h3 className="text-sm font-semibold text-cyan-300">
                            Traffic Data Source
                        </h3>

                        <p className="mt-1 text-xs leading-6 text-slate-400">
                            CIC-IDS2017 simulated sample ·{" "}
                            {traffic?.sample_records ?? 0} available records ·
                            Predictions generated by SentinelAI Random Forest.
                            This is dataset-driven simulation, not live packet capture.
                        </p>
                    </div>
                </section>

            </div>
        </main>
    );
}
