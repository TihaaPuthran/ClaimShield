import axios from 'axios'

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  'http://127.0.0.1:8000'

export async function submitClaim(formData) {
  if (!formData.get('user_id')) formData.append('user_id', 'USR-001')
  const response = await axios.post(
    `${API_BASE_URL}/claims/submit`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 90000,
    },
  )

  return response.data
}

export async function getClaimHistory(userId) {
  const response = await axios.get(`${API_BASE_URL}/users/${encodeURIComponent(userId)}/claims`)
  return response.data
}

export async function getClaimHistorySummary(userId) {
  const response = await axios.get(`${API_BASE_URL}/users/${encodeURIComponent(userId)}/claims/summary`)
  return response.data
}
