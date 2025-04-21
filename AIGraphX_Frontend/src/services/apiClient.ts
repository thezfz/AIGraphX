import axios from 'axios';

const apiClient = axios.create({
  baseURL: '/api/v1', // 假设使用代理转发到后端
  timeout: 10000, // 请求超时时间 10 秒
  headers: {
    'Content-Type': 'application/json',
  },
});

// 可以添加请求/响应拦截器 (可选)
// apiClient.interceptors.request.use(...)
// apiClient.interceptors.response.use(...)

export default apiClient; 