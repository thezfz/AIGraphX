import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0', // 允许外部访问
    port: 5173,      // 开发服务器端口
    proxy: {
      // 将 /api 开头的请求代理到后端容器
      // 'app' 是 compose.yml 中后端服务的名称
      '/api': {
        target: 'http://app:8000', // 后端服务地址和端口 (使用服务名 app)
        changeOrigin: true,       // 需要虚拟主机站点
        // secure: false,         // 如果后端是 https 但证书无效，可能需要
        // rewrite: (path) => path.replace(/^\/api/, ''), // 如果后端路径不包含 /api，取消注释此行
      },
    },
    // 解决 Podman + Vite HMR 问题
    // watch 选项应放在 server 内部
    watch: {
      usePolling: true,
    },
  },
}) 