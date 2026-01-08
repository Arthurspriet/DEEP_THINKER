/**
 * Server-Sent Events (SSE) utilities
 */

/**
 * Create an SSE connection for mission events
 * @param {string} missionId - Mission ID to subscribe to
 * @param {Object} handlers - Event handlers
 * @returns {EventSource} The event source connection
 */
export function subscribeMissionEvents(missionId, handlers = {}) {
  const eventSource = new EventSource(`/api/missions/${missionId}/events`)
  
  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      const eventType = data.type
      
      // Call specific handler if exists
      if (handlers[eventType]) {
        handlers[eventType](data.data, data.timestamp)
      }
      
      // Always call onMessage if provided
      if (handlers.onMessage) {
        handlers.onMessage(data)
      }
    } catch (e) {
      console.error('Failed to parse SSE message:', e)
    }
  }
  
  eventSource.onerror = (error) => {
    if (handlers.onError) {
      handlers.onError(error)
    }
  }
  
  eventSource.onopen = () => {
    if (handlers.onOpen) {
      handlers.onOpen()
    }
  }
  
  return eventSource
}

/**
 * Create an SSE connection for workflow events
 * @param {string} workflowId - Workflow ID to subscribe to
 * @param {Object} handlers - Event handlers
 * @returns {EventSource} The event source connection
 */
export function subscribeWorkflowEvents(workflowId, handlers = {}) {
  const eventSource = new EventSource(`/api/workflows/${workflowId}/events`)
  
  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      
      if (handlers.onMessage) {
        handlers.onMessage(data)
      }
    } catch (e) {
      console.error('Failed to parse SSE message:', e)
    }
  }
  
  eventSource.onerror = (error) => {
    if (handlers.onError) {
      handlers.onError(error)
    }
  }
  
  return eventSource
}

