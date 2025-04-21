import React, { Suspense } from 'react'
import ReactDOM from 'react-dom/client'
import { RouterProvider } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import router from './router'
import './index.css' // 引入 CSS (稍后创建)

// 创建一个 QueryClient 实例
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false, // 默认禁止窗口聚焦时自动重新获取
      retry: 1, // 默认重试1次
      staleTime: 5 * 60 * 1000, // 数据5分钟内视为新鲜
    },
  },
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <Suspense fallback={<div>加载中...</div>}>
        <RouterProvider router={router} />
      </Suspense>
    </QueryClientProvider>
  </React.StrictMode>,
) 