
"use client";

import { useState } from "react";
import Link from "next/link";
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  LoaderCircle,
  Play,
  RotateCcw,
  ShieldAlert,
  ShieldCheck,
} from "lucide-react";
import {
  predictNetworkTraffic,
  type PredictionResponse,
} from "@/lib/api";

type SampleType = "benign" | "attack";

interface SamplePayload {
  features: Record<string, number>;
}

export default function ThreatDetectionPage() {
  const [features, setFeatures] = useState<Record<string, number> | null>(null);
  const [sampleType, setSampleType] = useState<SampleType | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState("");

  async function loadSample(type: SampleType) {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`/samples/${type}.json`);

      if (!response.ok) {
        throw new Error("Could not load sample file.");
      }

      const data: SamplePayload = await response.json();

      if (
        !data.features ||
        Object.keys(data.features).length !== 70 ||
        Object.values(data.features).some(
          (value) => typeof value !== "number" || !Number.isFinite(value)
        )
      ) {
        throw new Error("Invalid sample. Expected 70 numeric features.");
      }

      setFeatures(data.features);
      setSampleType(type);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sample loading failed.");
    } finally {
      setLoading(false);
    }
  }

  async function runPrediction() {
    if (!features) return;

    setPredicting(true);
    setError("");
    setResult(null);

    try {
      const prediction = await predictNetworkTraffic({ features });
      setResult(prediction);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Prediction failed. Check the backend."
      );
    } finally {
      setPredicting(false);
    }
  }

  function reset() {
    setFeatures(null);
    setSampleType(null);
    setResult(null);
    setError("");
  }

  const isAttack = result?.prediction === "ATTACK";

  return (
    <main className="min-h-screen bg-[#080d17] px-5 py-8 text-[#e8edf7] sm:px-10">
      <div className="mx-auto max-w-6xl">
        <Link
          href="/"
          className="mb-8 inline-flex items-center gap-2 text-sm text-[#8995a9] hover:text-[#36d6a5]"
        >
          <ArrowLeft size={17} />
          Back to Dashboard
        </Link>

        <header className="mb-8 flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-[#36d6a5]/10 p-3">
                <ShieldCheck size={27} className="text-[#36d6a5]" />
              </div>
              <div>
                <h1 className="text-3xl font-bold">
                  AI Threat Detection
                </h1>
                <p className="mt-1 text-sm text-[#8995a9]">
                  Intelligent network traffic classification
                </p>
              </div>
            </div>
          </div>

          <span className="rounded-full border border-[#36d6a5]/30 bg-[#36d6a5]/10 px-4 py-2 text-xs font-semibold text-[#36d6a5]">
            Random Forest AI
          </span>
        </header>

        <section className="mb-6 grid gap-4 sm:grid-cols-3">
          {[
            { label: "AI Model", value: "Random Forest" },
            { label: "Input Features", value: "70 Features" },
            { label: "Classification", value: "BENIGN / ATTACK" },
          ].map((item) => (
            <div
              key={item.label}
              className="rounded-xl border border-[#202c40] bg-[#101827] p-5"
            >
              <Activity className="mb-3 text-[#36d6a5]" size={21} />
              <p className="text-xs text-[#8995a9]">{item.label}</p>
              <p className="mt-1 font-semibold">{item.value}</p>
            </div>
          ))}
        </section>

        <section className="grid gap-6 lg:grid-cols-2">
          <div className="rounded-2xl border border-[#202c40] bg-[#101827] p-6">
            <h2 className="text-lg font-semibold">
              Network Traffic Input
            </h2>
            <p className="mt-2 text-sm leading-6 text-[#8995a9]">
              Select a prepared CIC-IDS2017 network-flow sample.
              Each sample contains the 70 features required by the model.
            </p>

            <div className="mt-6 grid gap-3 sm:grid-cols-2">
              <button
                onClick={() => loadSample("benign")}
                disabled={loading || predicting}
                className="flex items-center justify-center gap-2 rounded-xl border border-[#36d6a5]/30 bg-[#36d6a5]/10 p-4 text-sm font-semibold text-[#36d6a5] hover:bg-[#36d6a5]/20 disabled:opacity-40"
              >
                <ShieldCheck size={19} />
                Load BENIGN Sample
              </button>

              <button
                onClick={() => loadSample("attack")}
                disabled={loading || predicting}
                className="flex items-center justify-center gap-2 rounded-xl border border-red-400/30 bg-red-400/10 p-4 text-sm font-semibold text-red-400 hover:bg-red-400/20 disabled:opacity-40"
              >
                <ShieldAlert size={19} />
                Load ATTACK Sample
              </button>
            </div>

            <div className="mt-6 rounded-xl border border-[#202c40] bg-[#080d17] p-5">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">
                  Input Status
                </span>
                <span className="text-xs text-[#36d6a5]">
                  {features ? "LOADED" : "WAITING"}
                </span>
              </div>

              {loading ? (
                <p className="mt-4 flex items-center gap-2 text-sm text-amber-400">
                  <LoaderCircle size={16} className="animate-spin" />
                  Loading sample...
                </p>
              ) : features ? (
                <div className="mt-4 space-y-2 text-sm">
                  <p className="flex justify-between">
                    <span className="text-[#8995a9]">Sample</span>
                    <strong className="uppercase">{sampleType}</strong>
                  </p>
                  <p className="flex justify-between">
                    <span className="text-[#8995a9]">Features</span>
                    <strong>{Object.keys(features).length} / 70</strong>
                  </p>
                </div>
              ) : (
                <p className="mt-4 text-sm text-[#8995a9]">
                  No sample selected.
                </p>
              )}
            </div>

            <div className="mt-5 flex gap-3">
              <button
                onClick={runPrediction}
                disabled={!features || predicting || loading}
                className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-[#36d6a5] p-4 text-sm font-bold text-[#08111c] hover:bg-[#57e4b9] disabled:cursor-not-allowed disabled:opacity-40"
              >
                {predicting ? (
                  <LoaderCircle size={18} className="animate-spin" />
                ) : (
                  <Play size={18} />
                )}
                {predicting ? "Analyzing..." : "Run AI Prediction"}
              </button>

              <button
                onClick={reset}
                disabled={predicting || loading}
                className="rounded-xl border border-[#202c40] px-4 text-[#8995a9] hover:bg-[#202c40] disabled:opacity-40"
                title="Reset"
              >
                <RotateCcw size={18} />
              </button>
            </div>

            {error && (
              <div className="mt-5 flex gap-3 rounded-xl border border-red-400/20 bg-red-400/10 p-4 text-sm text-red-300">
                <AlertTriangle size={19} className="shrink-0" />
                {error}
              </div>
            )}
          </div>

          <div className="rounded-2xl border border-[#202c40] bg-[#101827] p-6">
            <h2 className="text-lg font-semibold">Prediction Results</h2>
            <p className="mt-2 text-sm text-[#8995a9]">
              AI classification and probability analysis.
            </p>

            {!result ? (
              <div className="flex min-h-[350px] flex-col items-center justify-center text-center">
                <Activity size={42} className="mb-4 text-[#8995a9]" />
                <h3 className="font-semibold">Awaiting Prediction</h3>
                <p className="mt-2 max-w-xs text-sm leading-6 text-[#8995a9]">
                  Load a sample and run the AI model to see
                  the classification results.
                </p>
              </div>
            ) : (
              <div className="mt-6">
                <div
                  className={`rounded-2xl border p-7 text-center ${isAttack
                      ? "border-red-400/30 bg-red-400/5"
                      : "border-[#36d6a5]/30 bg-[#36d6a5]/5"
                    }`}
                >
                  {isAttack ? (
                    <ShieldAlert size={42} className="mx-auto mb-3 text-red-400" />
                  ) : (
                    <ShieldCheck size={42} className="mx-auto mb-3 text-[#36d6a5]" />
                  )}

                  <p className="text-xs uppercase tracking-widest text-[#8995a9]">
                    Detected Classification
                  </p>
                  <h3
                    className={`mt-2 text-3xl font-bold ${isAttack ? "text-red-400" : "text-[#36d6a5]"
                      }`}
                  >
                    {result.prediction}
                  </h3>
                  <p className="mt-2 text-xs text-[#8995a9]">
                    Prediction Code: {result.prediction_code}
                  </p>
                </div>

                <div className="mt-7 space-y-6">
                  {[
                    {
                      label: "Attack Probability",
                      value: result.attack_probability,
                      color: "bg-red-400",
                      text: "text-red-400",
                    },
                    {
                      label: "Benign Probability",
                      value: result.benign_probability,
                      color: "bg-[#36d6a5]",
                      text: "text-[#36d6a5]",
                    },
                  ].map((item) => (
                    <div key={item.label}>
                      <div className="mb-2 flex justify-between text-sm">
                        <span className="text-[#8995a9]">
                          {item.label}
                        </span>
                        <strong className={item.text}>
                          {(item.value * 100).toFixed(4)}%
                        </strong>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-[#202c40]">
                        <div
                          className={`h-full rounded-full ${item.color}`}
                          style={{
                            width: `${item.value * 100}%`,
                          }}
                        />
                      </div>
                    </div>
                  ))}

                  <div className="rounded-xl border border-[#202c40] bg-[#080d17] p-4">
                    <div className="flex justify-between">
                      <span className="text-sm text-[#8995a9]">
                        Prediction Confidence
                      </span>
                      <strong className="text-xl">
                        {(result.confidence * 100).toFixed(2)}%
                      </strong>
                    </div>
                  </div>

                  <p className="text-xs text-[#8995a9]">
                    Model: {result.model}
                  </p>
                </div>
              </div>
            )}
          </div>
        </section>

        <div className="mt-7 flex gap-3 rounded-xl border border-amber-400/20 bg-amber-400/5 p-4 text-sm leading-6 text-amber-200/80">
          <AlertTriangle size={19} className="shrink-0" />
          <p>
            <strong>Demo limitation:</strong> This page classifies
            prepared CIC-IDS2017 samples. It does not capture or
            monitor live network packets.
          </p>
        </div>

        <footer className="mt-8 border-t border-[#202c40] pt-5 text-center text-xs text-[#8995a9]">
          SentinelAI · AI-Powered Network Threat Classification
        </footer>
      </div>
    </main>
  );
}
