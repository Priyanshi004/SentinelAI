
import axios from "axios";

const API_BASE_URL =
    process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000,
    headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
    },
});

export interface HealthResponse {
    status: string;
    model_loaded: boolean;
    model: string;
    features_expected: number;
}

export interface PredictionResponse {
    prediction: "BENIGN" | "ATTACK";
    prediction_code: 0 | 1;
    attack_probability: number;
    benign_probability: number;
    confidence: number;
    model: string;
}

export interface PredictionRequest {
    features: Record<string, number>;
}

export interface TrafficFlow {
    flow_id: string;
    timestamp: string;
    destination_port: number;
    flow_duration: number;
    total_packets: number;
    total_bytes: number;
    prediction: "BENIGN" | "ATTACK";
    attack_probability: number;
    benign_probability: number;
    confidence: number;
    model: string;
    dataset_label: "BENIGN" | "ATTACK";
    source: string;
}

export interface TrafficStatus {
    is_running: boolean;
    total_flows: number;
    benign_flows: number;
    attack_flows: number;
    attack_rate: number;
    sample_records: number;
    recent_events: TrafficFlow[];
}

export interface ApiResponse<T> {
    message: string;
    data: T;
}

export const checkBackendHealth = async (): Promise<HealthResponse> => {
    const response = await api.get<HealthResponse>("/health");
    return response.data;
};

export const predictNetworkTraffic = async (
    payload: PredictionRequest
): Promise<PredictionResponse> => {
    const response = await api.post<PredictionResponse>(
        "/api/predict",
        payload
    );

    return response.data;
};

// Start traffic simulation
export const startTrafficSimulation = async (): Promise<TrafficStatus> => {
    const response = await api.post<ApiResponse<TrafficStatus>>(
        "/api/traffic/start"
    );

    return response.data.data;
};

// Stop traffic simulation
export const stopTrafficSimulation = async (): Promise<TrafficStatus> => {
    const response = await api.post<ApiResponse<TrafficStatus>>(
        "/api/traffic/stop"
    );

    return response.data.data;
};

// Get current traffic statistics
export const getTrafficStatus = async (): Promise<TrafficStatus> => {
    const response = await api.get<
        ApiResponse<TrafficStatus> | TrafficStatus
    >("/api/traffic/status");

    if ("data" in response.data) {
        return response.data.data;
    }

    return response.data;
};

// Process the next simulated traffic flow
export const getNextTrafficFlow = async (): Promise<TrafficFlow> => {
    const response = await api.get<ApiResponse<TrafficFlow>>(
        "/api/traffic/next"
    );

    return response.data.data;
};

export default api;
