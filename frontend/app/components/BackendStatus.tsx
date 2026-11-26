
"use client";

import { useEffect, useState } from "react";
import { Activity, AlertCircle, CheckCircle2, LoaderCircle } from "lucide-react";
import { checkBackendHealth } from "@/lib/api";

type ConnectionStatus = "checking" | "connected" | "disconnected";

export default function BackendStatus() {
    const [status, setStatus] = useState<ConnectionStatus>("checking");
    const [model, setModel] = useState("Checking backend...");
    const [features, setFeatures] = useState(0);

    useEffect(() => {
        let mounted = true;

        const checkConnection = async () => {
            try {
                const health = await checkBackendHealth();

                if (!mounted) return;

                setStatus(
                    health.status === "healthy" && health.model_loaded
                        ? "connected"
                        : "disconnected"
                );
                setModel(health.model);
                setFeatures(health.features_expected);
            } catch {
                if (!mounted) return;

                setStatus("disconnected");
                setModel("Backend unavailable");
                setFeatures(0);
            }
        };

        checkConnection();

        const interval = setInterval(checkConnection, 15000);

        return () => {
            mounted = false;
            clearInterval(interval);
        };
    }, []);

    return (
        <div className="flex items-center gap-2 rounded-lg border border-[#202c40] bg-[#101827] px-3 py-2">
            {status === "checking" && (
                <LoaderCircle size={15} className="animate-spin text-amber-400" />
            )}

            {status === "connected" && (
                <CheckCircle2 size={15} className="text-[#36d6a5]" />
            )}

            {status === "disconnected" && (
                <AlertCircle size={15} className="text-red-400" />
            )}

            <div>
                <p
                    className={`text-[11px] font-medium ${status === "connected"
                        ? "text-[#36d6a5]"
                        : status === "disconnected"
                            ? "text-red-400"
                            : "text-amber-400"
                        }`}
                >
                    {status === "checking"
                        ? "Checking backend"
                        : status === "connected"
                            ? "Backend Connected"
                            : "Backend Disconnected"}
                </p>

                <p className="text-[9px] text-[#8995a9]">
                    {status === "connected"
                        ? `${model} | ${features} features`
                        : status === "disconnected"
                            ? "Check FastAPI server"
                            : "Connecting..."}
                </p>
            </div>

            <Activity size={14} className="ml-1 text-[#8995a9]" />
        </div>
    );
}
