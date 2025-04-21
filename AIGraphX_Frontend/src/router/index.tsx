import React, { lazy } from 'react';
import { createBrowserRouter, RouteObject } from 'react-router-dom';
import App from '../App';

// 导入页面组件（这些文件稍后会创建）
const SearchPage = lazy(() => import('../pages/SearchPage'));
const PaperDetailPage = lazy(() => import('../pages/PaperDetailPage'));
const ModelDetailPage = lazy(() => import('../pages/ModelDetailPage'));

// 路由配置
const routes: RouteObject[] = [
  {
    path: '/',
    element: <App />,
    children: [
      {
        index: true,
        element: <SearchPage />,
      },
      {
        path: 'papers/:pwcId',
        element: <PaperDetailPage />,
      },
      {
        path: 'models/:modelId',
        element: <ModelDetailPage />,
      },
    ],
  },
];

// 创建路由器
const router = createBrowserRouter(routes);

export default router; 