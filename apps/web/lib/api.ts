import type { ApiErrorEnvelope } from "@/types/api";

export const API_URL =
  process.env.NEXT_PUBLIC_RAGLAB_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

export const DEMO_MODE = process.env.NEXT_PUBLIC_RAGLAB_DEMO_MODE === "true";
export const API_KEY_STORAGE = "raglab.api-key";

export class RAGLabApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly errorType: string,
  ) {
    super(message);
    this.name = "RAGLabApiError";
  }
}

export function storedApiKey(): string {
  return typeof window === "undefined" ? "" : sessionStorage.getItem(API_KEY_STORAGE) ?? "";
}

export function storeApiKey(value: string): void {
  if (typeof window === "undefined") return;
  const clean = value.trim();
  if (clean) sessionStorage.setItem(API_KEY_STORAGE, clean);
  else sessionStorage.removeItem(API_KEY_STORAGE);
  window.dispatchEvent(new Event("raglab:credentials"));
}

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const apiKey = storedApiKey();
  if (apiKey) headers.set("Authorization", `Bearer ${apiKey}`);
  if (init.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");
  headers.set("Accept", "application/json");

  const response = await fetch(`${API_URL}${path}`, { ...init, headers, cache: "no-store" });
  if (!response.ok) {
    let envelope: ApiErrorEnvelope = {};
    try {
      envelope = (await response.json()) as ApiErrorEnvelope;
    } catch {
      // The safe fallback below intentionally ignores non-JSON provider bodies.
    }
    throw new RAGLabApiError(
      envelope.error?.message ?? `RAGLab returned HTTP ${response.status}`,
      response.status,
      envelope.error?.type ?? "RequestFailed",
    );
  }
  return (await response.json()) as T;
}

export async function apiText(path: string): Promise<string> {
  const response = await fetch(`${API_URL}${path}`, { cache: "no-store" });
  if (!response.ok) throw new RAGLabApiError(`RAGLab returned HTTP ${response.status}`, response.status, "RequestFailed");
  return response.text();
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "The local service did not respond.";
}
