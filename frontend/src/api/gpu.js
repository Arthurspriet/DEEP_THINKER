/**
 * GPU & Resources API client
 */

/**
 * Get GPU statistics
 */
export async function getGpuStats() {
  const response = await fetch('/api/gpu/stats')
  if (!response.ok) {
    throw new Error(`Failed to fetch GPU stats: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Get overall resource status
 */
export async function getResourceStatus() {
  const response = await fetch('/api/resources/status')
  if (!response.ok) {
    throw new Error(`Failed to fetch resource status: ${response.statusText}`)
  }
  return response.json()
}

