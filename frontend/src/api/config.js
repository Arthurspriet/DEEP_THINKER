/**
 * Configuration API client
 */

const API_BASE = '/api/config'

/**
 * Get available models from Ollama
 */
export async function getModels() {
  const response = await fetch(`${API_BASE}/models`)
  if (!response.ok) {
    throw new Error(`Failed to fetch models: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Get council configurations
 */
export async function getCouncils() {
  const response = await fetch(`${API_BASE}/councils`)
  if (!response.ok) {
    throw new Error(`Failed to fetch councils: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Get agent configurations
 */
export async function getAgents() {
  const response = await fetch(`${API_BASE}/agents`)
  if (!response.ok) {
    throw new Error(`Failed to fetch agents: ${response.statusText}`)
  }
  return response.json()
}

