import api from '../api/axios';

/**
 * Upload document and get summary
 */
export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append('document', file);

  const response = await api.post('/documents/summarize/', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
}

/**
 * Send chat message
 */
export async function sendChatMessage(sessionId, message) {
  const response = await api.post('/documents/chat/', {
    session_id: sessionId,
    message: message,
  });
  
  return response.data;
}

/**
 * Get chat history for a session
 */
export async function getChatHistory(sessionId) {
  const response = await api.get(`/documents/sessions/${sessionId}/`);
  return response.data;
}

/**
 * Get user's document sessions
 */
export async function getUserSessions() {
  const response = await api.get('/documents/sessions/');
  return response.data;
}

