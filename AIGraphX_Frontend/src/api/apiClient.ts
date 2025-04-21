import axios from 'axios';

// Create an Axios instance
const apiClient = axios.create({
  // The base URL will be handled by the Vite proxy for /api requests
  // baseURL: 'http://localhost:8000', // Set this if not using proxy or for other requests
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