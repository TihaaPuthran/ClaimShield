import axios from 'axios'

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.DEV ? 'http://127.0.0.1:8000' : '')
).replace(/\/$/, '')

export async function submitClaim(formData) {
  if (!formData.get('user_id')) formData.append('user_id', 'USR-001')
  const fullRequestUrl = `${API_BASE_URL}/claims/submit`
  if (!API_BASE_URL) throw new Error('VITE_API_BASE_URL is not configured for production.')
  console.log('API Base URL:', import.meta.env.VITE_API_BASE_URL)
  console.log('Submitting to:', fullRequestUrl)
  const response = await axios.post(
    fullRequestUrl,
    formData,
    {
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
