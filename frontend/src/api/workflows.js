/**
 * Workflow API client (legacy mode)
 */

const API_BASE = '/api/workflows'

/**
 * Run a new workflow
 * @param {Object} workflowData - Workflow configuration
 */
export async function runWorkflow(workflowData) {
  const response = await fetch(`${API_BASE}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(workflowData),
  })
  if (!response.ok) {
    throw new Error(`Failed to start workflow: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Get workflow status
 * @param {string} workflowId - Workflow ID
 */
export async function getWorkflow(workflowId) {
  const response = await fetch(`${API_BASE}/${workflowId}`)
  if (!response.ok) {
    throw new Error(`Failed to fetch workflow: ${response.statusText}`)
  }
  return response.json()
}

