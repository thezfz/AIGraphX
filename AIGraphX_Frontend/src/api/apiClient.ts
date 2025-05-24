import axios from 'axios';

// Create an Axios instance
const apiClient = axios.create({
  baseURL: '/api/v1', // Set base URL from services/apiClient.ts
  timeout: 10000, // Set timeout from services/apiClient.ts
  headers: {
    'Content-Type': 'application/json',
  },
});

// You can add interceptors here if needed (e.g., for error handling or adding auth tokens)
/*
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle errors globally
    console.error('API call error:', error);
    return Promise.reject(error);
  }
);
*/

export default apiClient; 