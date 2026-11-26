"use client";

import { useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  LoaderCircle,
  Play,
  RotateCcw,
  Shield,
  ShieldAlert,
  ShieldCheck,
  Upload,
} from "lucide-react";
import {
  predictNetworkTraffic,
  type PredictionResponse,
} from "@/lib/api";

type SampleType = "benign" | "attack";

export default function ThreatDetectionPage() {
  const [features, setFeatures] = useState<Record<string, number> | null>(null);
  const [sampleType, setSampleType] = useState<SampleType | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const loadSample = async (type: SampleType) => {
    try {
      setError("");
      setResult(null);
      setFeatures(null);
      setSampleType(type);

      const response = await fetch(`/samples/${type}.json`);

      if (!response.ok) {
        throw new Error(`Unable to load ${type} sample.`);
      }

      const data = await response.json();
      const sampleFeatures = data.features ?? data;

      if (
        typeof sampleFeatures !== "object" ||
        sampleFeatures === null ||
        Object.keys(sampleFeatures).length !== 70
      ) {
        throw new Error("The sample must contain exactly 70 features.");
      }

      const values = Object.values(sampleFeatures);

      if (
        !values.every(
          (value) => typeof value === "number" && Number.isFinite(value)
        )
      ) {
        throw new Error("The sample contains invalid or non-numeric features.");
      }

      setFeatures(sampleFeatures as Record<string, number>);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Something went wrong."
      );
      setSampleType(null);
    }
  };

  const runDetection = async () => {
    if (!features) {
      setError("Please load a sample before running detection.");
      return;
    }

    try {
      setLoading(true);
      setError("");
      setResult(null);

      const prediction = await predictNetworkTraffic({ features });
      setResult(prediction);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Prediction request failed.";

      setError(
        message.includes("Network Error")
          ? "Cannot connect to the backend. Make sure FastAPI is running on port 8000."
          : message
      );
    } finally {
      setLoading(false);
    }
  };

  const resetDetection = () => {
    setFeatures(null);
    setSampleType(null);
    setResult(null);
    setError("");
    setLoading(false);
  };

  const isAttack = result?.prediction === "ATTACK";

  return (
    <main className="min-h-screen space-y-8 bg-[#080d19] p-5 text-slate-100 md:p-8">
      <section className="flex flex-col justify-between gap-4 md:flex-row md:items-center">
        <div>
          <div className="mb-2 flex items-center gap-2 text-sm text-cyan-400">
            <Shield className="h-4 w-4" />
            Security Operations Center
          </div>
          <h1 className="text-3xl font-bold tracking-tight">
            Threat Detection
          </h1>
          <p className="mt-2 text-sm text-slate-400">
            Analyze network flow data using the SentinelAI Random Forest model.
          </p>
        </div>

        <div className="flex items-center gap-2 rounded-xl border border-emerald-500/20 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-400">
          <Activity className="h-4 w-4" />
          ML Engine Integration
        </div>
      </section>

      <section className="rounded-2xl border border-amber-500/20 bg-amber-500/5 p-4">
        <div className="flex items-start gap-3">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-400" />
          <div>
            <h2 className="text-sm font-semibold text-amber-300">
              Dataset and demonstration notice
            </h2>
            <p className="mt-1 text-sm leading-6 text-slate-400">
              This page analyzes prepared CIC-IDS2017 sample network-flow
              records. It demonstrates model inference and does not capture
              or monitor your device&apos;s live network packets.
            </p>
          </div>
        </div>
      </section>

      <section className="grid gap-6 lg:grid-cols-[1fr_1fr]">
        <div className="rounded-2xl border border-slate-800 bg-[#101827] p-6">
          <div className="mb-6 flex items-center gap-3">
            <div className="rounded-xl bg-cyan-500/10 p-3 text-cyan-400">
              <Upload className="h-5 w-5" />
            </div>
            <div>
              <h2 className="font-semibold">Network Flow Input</h2>
              <p className="text-xs text-slate-400">
                Choose a prepared test sample
              </p>
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <button
              onClick={() => loadSample("benign")}
              disabled={loading}
              className={`rounded-xl border p-4 text-left transition ${
                sampleType === "benign"
                  ? "border-emerald-500 bg-emerald-500/10"
                  : "border-slate-700 bg-slate-900/60 hover:border-emerald-500/50"
              }`}
            >
              <ShieldCheck className="mb-3 h-6 w-6 text-emerald-400" />
              <p className="font-semibold">BENIGN Sample</p>
              <p className="mt-1 text-xs text-slate-400">
                Normal network traffic
              </p>
            </button>

            <button
              onClick={() => loadSample("attack")}
              disabled={loading}
              className={`rounded-xl border p-4 text-left transition ${
                sampleType === "attack"
                  ? "border-red-500 bg-red-500/10"
                  : "border-slate-700 bg-slate-900/60 hover:border-red-500/50"
              }`}
            >
              <ShieldAlert className="mb-3 h-6 w-6 text-red-400" />
              <p className="font-semibold">ATTACK Sample</p>
              <p className="mt-1 text-xs text-slate-400">
                Malicious network traffic
              </p>
            </button>
          </div>

          <div className="mt-5 rounded-xl border border-slate-800 bg-[#0b1220] p-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-400">Input status</span>
              <span
                className={`text-xs font-medium ${
                  features ? "text-emerald-400" : "text-slate-500"
                }`}
              >
                {features ? "Sample loaded" : "Waiting for input"}
              </span>
            </div>

            <div className="mt-3 flex items-center justify-between">
              <span className="text-sm text-slate-400">Features loaded</span>
              <span className="font-mono text-sm text-slate-200">
                {features ? `${Object.keys(features).length} / 70` : "0 / 70"}
              </span>
            </div>

            <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-800">
              <div
                className="h-full rounded-full bg-cyan-400 transition-all"
                style={{ width: features ? "100%" : "0%" }}
              />
            </div>
          </div>

          <div className="mt-5 flex flex-col gap-3 sm:flex-row">
            <button
              onClick={runDetection}
              disabled={!features || loading}
              className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-cyan-500 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {loading ? (
                <LoaderCircle className="h-4 w-4 animate-spin" />
              ) : (
                <Play className="h-4 w-4" />
              )}
              {loading ? "Analyzing Traffic..." : "Run Detection"}
            </button>

            <button
              onClick={resetDetection}
              disabled={loading}
              className="flex items-center justify-center gap-2 rounded-xl border border-slate-700 px-5 py-3 text-sm text-slate-300 transition hover:bg-slate-800 disabled:opacity-40"
            >
              <RotateCcw className="h-4 w-4" />
              Reset
            </button>
          </div>

          {error && (
            <div className="mt-4 flex items-start gap-2 rounded-xl border border-red-500/20 bg-red-500/10 p-3 text-sm text-red-300">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              {error}
            </div>
          )}
        </div>

        <div className="rounded-2xl border border-slate-800 bg-[#101827] p-6">
          <div className="mb-6 flex items-center gap-3">
            <div className="rounded-xl bg-violet-500/10 p-3 text-violet-400">
              <Activity className="h-5 w-5" />
            </div>
            <div>
              <h2 className="font-semibold">Detection Results</h2>
              <p className="text-xs text-slate-400">
                Machine learning prediction output
              </p>
            </div>
          </div>

          {!result && !loading && (
            <div className="flex min-h-[290px] flex-col items-center justify-center rounded-xl border border-dashed border-slate-700 bg-[#0b1220] p-6 text-center">
              <Shield className="mb-4 h-12 w-12 text-slate-600" />
              <h3 className="font-medium text-slate-300">
                No prediction available
              </h3>
              <p className="mt-2 max-w-xs text-sm text-slate-500">
                Load a network sample and run detection to view the model
                prediction.
              </p>
            </div>
          )}

          {loading && (
            <div className="flex min-h-[290px] flex-col items-center justify-center rounded-xl bg-[#0b1220]">
              <LoaderCircle className="mb-4 h-10 w-10 animate-spin text-cyan-400" />
              <p className="font-medium text-slate-200">
                Analyzing network traffic
              </p>
              <p className="mt-2 text-sm text-slate-500">
                Waiting for the ML model response...
              </p>
            </div>
          )}

          {result && (
            <div className="space-y-5">
              <div
                className={`rounded-xl border p-5 ${
                  isAttack
                    ? "border-red-500/30 bg-red-500/10"
                    : "border-emerald-500/30 bg-emerald-500/10"
                }`}
              >
                <div className="flex items-center gap-3">
                  {isAttack ? (
                    <ShieldAlert className="h-9 w-9 text-red-400" />
                  ) : (
                    <CheckCircle2 className="h-9 w-9 text-emerald-400" />
                  )}
                  <div>
                    <p className="text-xs uppercase tracking-wider text-slate-400">
                      Prediction
                    </p>
                    <h3
                      className={`text-2xl font-bold ${
                        isAttack ? "text-red-400" : "text-emerald-400"
                      }`}
                    >
                      {result.prediction}
                    </h3>
                  </div>
                </div>
                <p className="mt-3 text-sm text-slate-400">
                  {isAttack
                    ? "The model classified this network flow as potentially malicious."
                    : "The model classified this network flow as benign traffic."}
                </p>
              </div>

              <div className="space-y-4">
                <ProbabilityBar
                  label="Attack Probability"
                  value={result.attack_probability}
                  color="bg-red-400"
                />
                <ProbabilityBar
                  label="Benign Probability"
                  value={result.benign_probability}
                  color="bg-emerald-400"
                />
                <ProbabilityBar
                  label="Prediction Confidence"
                  value={result.confidence}
                  color="bg-cyan-400"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-xl bg-[#0b1220] p-4">
                  <p className="text-xs text-slate-500">Prediction Code</p>
                  <p className="mt-1 font-mono text-lg font-semibold">
                    {result.prediction_code}
                  </p>
                </div>
                <div className="rounded-xl bg-[#0b1220] p-4">
                  <p className="text-xs text-slate-500">Model</p>
                  <p className="mt-1 text-sm font-semibold text-slate-200">
                    {result.model}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      <section className="rounded-2xl border border-slate-800 bg-[#101827] p-5">
        <h2 className="font-semibold">Detection Pipeline</h2>
        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          {[
            {
              number: "01",
              title: "Input Features",
              description: "Load 70 numeric network-flow features.",
            },
            {
              number: "02",
              title: "Random Forest",
              description: "Send the features to the FastAPI prediction endpoint.",
            },
            {
              number: "03",
              title: "Classification",
              description: "Display BENIGN or ATTACK with probability scores.",
            },
          ].map((step) => (
            <div
              key={step.number}
              className="rounded-xl border border-slate-800 bg-[#0b1220] p-4"
            >
              <p className="font-mono text-sm text-cyan-400">{step.number}</p>
              <h3 className="mt-2 font-semibold">{step.title}</h3>
              <p className="mt-1 text-sm leading-6 text-slate-400">
                {step.description}
              </p>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}

function ProbabilityBar({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  const percentage = Math.max(0, Math.min(100, value * 100));

  return (
    <div>
      <div className="mb-2 flex justify-between gap-3 text-sm">
        <span className="text-slate-400">{label}</span>
        <span className="font-mono font-medium text-slate-200">
          {percentage.toFixed(2)}%
        </span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-800">
        <div
          className={`h-full rounded-full transition-all ${color}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}
