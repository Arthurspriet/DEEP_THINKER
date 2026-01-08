/**
 * Agent monitoring API client
 */

const API_BASE = '/api/agents'

/**
 * Get status of all agents
 */
export async function getAgentsStatus() {
  const response = await fetch(`${API_BASE}/status`)
  if (!response.ok) {
    throw new Error(`Failed to fetch agent status: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Get traces for a specific agent
 * @param {string} agentName - Agent name
 * @param {number} limit - Max traces to fetch
 * @param {string|null} missionId - Optional mission filter
 */
export async function getAgentTraces(agentName, limit = 50, missionId = null) {
  let url = `${API_BASE}/${agentName}/traces?limit=${limit}`
  if (missionId) {
    url += `&mission_id=${missionId}`
  }
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to fetch agent traces: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Get aggregated metrics for all agents
 */
export async function getAgentMetrics() {
  const response = await fetch(`${API_BASE}/metrics`)
  if (!response.ok) {
    throw new Error(`Failed to fetch agent metrics: ${response.statusText}`)
  }
  return response.json()
}

