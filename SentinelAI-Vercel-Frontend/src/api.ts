export type Action = 'ALLOW' | 'MASK' | 'BLOCK'
export type PolicyConfig = {
  block_on_injection: boolean
  block_on_secrets: boolean
  mask_pii: boolean
  enforce_document_permissions: boolean
}
export type AuditLog = {
  id: string
  timestamp: string
  action: Action
  risk_score: number
  detected_types: string[]
  sanitized_preview: string
  user_role: string
  blocked_reason?: string | null
}
export type Metrics = {
  total_requests: number
  blocked_count: number
  masked_count: number
  allowed_count: number
  leak_type_distribution: Record<string, number>
}
export type ScanEntity = {
  entity_type: string
  label: string
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
  start: number
  end: number
}
export type ScanResponse = {
  status: string
  action: Action
  sanitized_prompt: string
  risk_score: number
  entities_detected: ScanEntity[]
  injection_detected: boolean
  permission_granted: boolean
  blocked_reason?: string | null
  llm_response?: string | null
}

const configuredBase = (import.meta.env.VITE_API_BASE_URL || '').trim().replace(/\/$/, '')

function endpoint(path: string): string {
  if (import.meta.env.PROD && !configuredBase) {
    throw new Error('Backend URL is not configured. In Vercel, set VITE_API_BASE_URL to your deployed FastAPI service URL, then redeploy.')
  }
  return `${configuredBase}/api${path}`
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = endpoint(path)
  let response: Response
  try {
    response = await fetch(url, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...init?.headers },
    })
  } catch {
    throw new Error('Could not reach the backend. Check VITE_API_BASE_URL, confirm the API is deployed over HTTPS, and allow this Vercel domain in backend CORS settings.')
  }
  if (!response.ok) {
    const detail = await response.text().catch(() => '')
    throw new Error(detail || `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string; service: string }>('/health'),
  metrics: () => request<Metrics>('/logs/metrics'),
  logs: (limit = 50) => request<AuditLog[]>(`/logs?limit=${limit}`),
  policies: () => request<PolicyConfig>('/rules'),
  savePolicies: (config: PolicyConfig) => request<PolicyConfig>('/rules', { method: 'POST', body: JSON.stringify(config) }),
  scan: (payload: { prompt: string; user_context: { user_id: string; role: string; department: string }; document_context: { doc_id: string; title: string; required_clearance: string } }) =>
    request<ScanResponse>('/gateway/scan', { method: 'POST', body: JSON.stringify(payload) }),
}
