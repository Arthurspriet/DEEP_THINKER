/**
 * Configuration API client
 * 
 * Provides access to model metadata, performance grades, and configurations.
 */

const API_BASE = '/api/config'

/**
 * Get available models with enhanced metadata and performance grades.
 * 
 * Returns models with:
 * - tier: reasoning/large/medium/small/embedding
 * - capabilities: scoring for reasoning, coding, etc.
 * - performance_grade: excellent/good/fair/poor/unknown
 * - is_known: true if in static registry, false if auto-discovered
 */
export async function getModels() {
  const response = await fetch(`${API_BASE}/models`)
  if (!response.ok) {
    throw new Error(`Failed to fetch models: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Get simple model list (legacy format without performance data)
 */
export async function getModelsSimple() {
  const response = await fetch(`${API_BASE}/models/simple`)
  if (!response.ok) {
    throw new Error(`Failed to fetch models: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Get detailed model statistics from mission history
 */
export async function getModelStats() {
  const response = await fetch(`${API_BASE}/models/stats`)
  if (!response.ok) {
    throw new Error(`Failed to fetch model stats: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Force refresh of model discovery
 */
export async function refreshModels() {
  const response = await fetch(`${API_BASE}/models/refresh`, {
    method: 'POST'
  })
  if (!response.ok) {
    throw new Error(`Failed to refresh models: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Get models filtered by tier
 * @param {string} tier - One of: reasoning, large, medium, small, embedding
 */
export async function getModelsByTier(tier) {
  const models = await getModels()
  return models.filter(m => m.tier === tier && m.is_available)
}

/**
 * Get models sorted by performance grade
 */
export async function getModelsByPerformance() {
  const models = await getModels()
  const gradeOrder = { excellent: 0, good: 1, fair: 2, poor: 3, unknown: 4 }
  return models
    .filter(m => m.is_available)
    .sort((a, b) => gradeOrder[a.performance_grade] - gradeOrder[b.performance_grade])
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

