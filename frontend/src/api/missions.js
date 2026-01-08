/**
 * Mission API client
 */

const API_BASE = '/api/missions'

/**
 * Fetch all missions
 * @param {string|null} status - Optional status filter
 */
export async function getMissions(status = null) {
  const url = status ? `${API_BASE}?status=${status}` : API_BASE
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to fetch missions: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Create a new mission
 * @param {Object} missionData - Mission creation data
 */
export async function createMission(missionData) {
  const response = await fetch(API_BASE, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(missionData),
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(error.detail || 'Failed to create mission')
  }
  return response.json()
}

/**
 * Get mission details
 * @param {string} missionId - Mission ID
 */
export async function getMission(missionId) {
  const response = await fetch(`${API_BASE}/${missionId}`)
  if (!response.ok) {
    throw new Error(`Failed to fetch mission: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Get mission status
 * @param {string} missionId - Mission ID
 */
export async function getMissionStatus(missionId) {
  const response = await fetch(`${API_BASE}/${missionId}/status`)
  if (!response.ok) {
    throw new Error(`Failed to fetch mission status: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Resume a mission
 * @param {string} missionId - Mission ID
 */
export async function resumeMission(missionId) {
  const response = await fetch(`${API_BASE}/${missionId}/resume`, {
    method: 'POST',
  })
  if (!response.ok) {
    throw new Error(`Failed to resume mission: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Abort a mission
 * @param {string} missionId - Mission ID
 * @param {string} reason - Abort reason
 */
export async function abortMission(missionId, reason = 'User requested abort') {
  const response = await fetch(`${API_BASE}/${missionId}/abort?reason=${encodeURIComponent(reason)}`, {
    method: 'POST',
  })
  if (!response.ok) {
    throw new Error(`Failed to abort mission: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Get mission logs
 * @param {string} missionId - Mission ID
 * @param {number} limit - Max logs to fetch
 * @param {number} offset - Offset for pagination
 */
export async function getMissionLogs(missionId, limit = 100, offset = 0) {
  const response = await fetch(`${API_BASE}/${missionId}/logs?limit=${limit}&offset=${offset}`)
  if (!response.ok) {
    throw new Error(`Failed to fetch logs: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Get mission artifacts
 * @param {string} missionId - Mission ID
 * @param {string|null} phase - Optional phase filter
 */
export async function getMissionArtifacts(missionId, phase = null) {
  const url = phase 
    ? `${API_BASE}/${missionId}/artifacts?phase=${encodeURIComponent(phase)}`
    : `${API_BASE}/${missionId}/artifacts`
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to fetch artifacts: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Get mission deliverables
 * @param {string} missionId - Mission ID
 */
export async function getMissionDeliverables(missionId) {
  const response = await fetch(`${API_BASE}/${missionId}/deliverables`)
  if (!response.ok) {
    throw new Error(`Failed to fetch deliverables: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Get mission alignment data
 * @param {string} missionId - Mission ID
 * @returns {Promise<Object>} Alignment data including trajectory, actions, and summary
 * 
 * Response shape:
 * {
 *   mission_id: string,
 *   enabled: boolean,
 *   north_star: { goal_id, intent_summary, ... } | null,
 *   trajectory: [{ t, phase_name, alignment_score, drift_delta, cusum_neg, warning, triggered, timestamp }],
 *   actions_taken: [{ action, t, phase_name, timestamp, consecutive_triggers }],
 *   summary: { current_alignment, total_points, trigger_count, warning_count, actions_count, status }
 * }
 */
export async function getMissionAlignment(missionId) {
  const response = await fetch(`${API_BASE}/${missionId}/alignment`)
  if (!response.ok) {
    // Return null for 404 (alignment disabled or no data)
    if (response.status === 404) {
      return null
    }
    throw new Error(`Failed to fetch alignment: ${response.statusText}`)
  }
  return response.json()
}

