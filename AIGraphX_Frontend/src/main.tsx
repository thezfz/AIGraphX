import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx' // 我们稍后会创建 App.tsx
import './index.css' // 引入 CSS (稍后创建)

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
) 