import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { useModelDetail } from '../api/apiQueries'; // 引入 Hook
import Spinner from '../components/common/Spinner'; // 引入 Spinner
import { ArrowDownTrayIcon, HeartIcon } from '@heroicons/react/24/outline'; // 引入图标

const ModelDetailPage: React.FC = () => {
  const { modelId } = useParams<{ modelId: string }>();
  const [showDebugInfo, setShowDebugInfo] = useState(false); // 添加状态控制调试信息显示

  // 使用 React Query Hook 获取数据
  const {
    data: model, // API 返回的数据结构可能与之前的 ModelDetail 不同
    isLoading,
    isError,
    error,
  } = useModelDetail(modelId); // 传入 modelId

  // 添加调试日志
  useEffect(() => {
    if (model) {
      console.log('Model detail received in component:', model);
      console.log('Model fields:', Object.keys(model));
    }
  }, [model]);

  // 如果没有 modelId，直接显示错误信息
  if (!modelId) {
    return (
      <div className="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded relative" role="alert">
        <strong className="font-bold">无效请求: </strong>
        <span className="block sm:inline">未提供模型 ID。</span>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex justify-center items-center py-20">
        <Spinner /> {/* 使用 Spinner 组件 */}
        <p className="ml-2 text-gray-600">正在加载模型 {modelId} 的详情...</p>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
        <strong className="font-bold">获取模型详情出错: </strong>
        <span className="block sm:inline">{error instanceof Error ? error.message : JSON.stringify(error)}</span>
        <div className="mt-2">
          <p className="text-sm">请求的模型 ID: {modelId}</p>
          <p className="text-sm">您可以尝试刷新页面或返回搜索。</p>
        </div>
      </div>
    );
  }

  if (!model) {
    return (
      <div className="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded relative" role="alert">
        <strong className="font-bold">未找到: </strong>
        <span className="block sm:inline">无法找到 ID 为 {modelId} 的模型。</span>
        <div className="mt-2">
          <p className="text-sm">可能的原因：</p>
          <ul className="list-disc list-inside text-sm">
            <li>模型 ID 不存在</li>
            <li>数据库中没有该模型的信息</li>
            <li>服务器响应格式与预期不符</li>
          </ul>
        </div>
      </div>
    );
  }

  // 安全获取属性的辅助函数
  const safeRender = (render: () => React.ReactNode) => {
    try {
      return render();
    } catch (error) {
      console.error('Render error:', error);
      return <span className="text-red-500">渲染错误</span>;
    }
  };

  // --- 渲染从 API 获取的模型数据 ---
  // 注意：这里的字段需要匹配 useModelDetail 返回的 ModelDetailResponse 类型
  try {
    return (
      <div className="bg-white rounded-lg shadow-md overflow-hidden"> {/* 添加 overflow-hidden 防止内容溢出 */} 
        {/* Header Section */}
        <div className="p-6 bg-gradient-to-r from-blue-50 to-indigo-50"> {/* 添加渐变背景 */} 
          <h1 className="text-2xl md:text-3xl font-bold text-gray-800 mb-1 break-all">
            {safeRender(() => model.model_id)}
          </h1>
          {model.author && (
            <p className="text-md text-gray-600">由 {safeRender(() => model.author)} 创建</p>
          )}
        </div>

        {/* Main Content Section */}
        <div className="p-6">
          {/* Metadata and Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            {/* Left Column: Metadata */}
            <div className="md:col-span-2 space-y-4"> {/* 使用 space-y 控制间距 */}
              {model.pipeline_tag && (
                <div>
                  <h3 className="text-sm font-medium text-gray-500">任务类型</h3>
                  <p className="text-lg text-gray-800">{safeRender(() => model.pipeline_tag)}</p>
                </div>
              )}
              {model.library_name && (
                <div>
                  <h3 className="text-sm font-medium text-gray-500">库</h3>
                  <p className="text-lg text-gray-800">{safeRender(() => model.library_name)}</p>
                </div>
              )}
              {model.last_modified && (
                <div>
                  <h3 className="text-sm font-medium text-gray-500">最后更新</h3>
                  <p className="text-lg text-gray-800">
                    {safeRender(() => model.last_modified ? new Date(model.last_modified).toLocaleDateString('zh-CN', { year: 'numeric', month: 'long', day: 'numeric' }) : '未知')} {/* 改进日期格式 */} 
                  </p>
                </div>
              )}
            </div>

            {/* Right Column: Stats */}
            <div className="bg-gray-50 rounded-lg p-4 space-y-3"> {/* 使用 space-y */} 
              <h2 className="text-lg font-semibold text-gray-700 mb-2">统计信息</h2>
              {model.downloads !== undefined && model.downloads !== null && (
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600 inline-flex items-center">
                    <ArrowDownTrayIcon className="mr-1.5 h-5 w-5 text-gray-400" aria-hidden="true" />
                    下载次数
                  </span>
                  <span className="font-medium text-gray-800">
                    {safeRender(() => model.downloads?.toLocaleString() || '0')}
                  </span>
                </div>
              )}
              {model.likes !== undefined && model.likes !== null && (
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600 inline-flex items-center">
                    <HeartIcon className="mr-1.5 h-5 w-5 text-gray-400" aria-hidden="true" />
                    点赞数
                  </span>
                  <span className="font-medium text-gray-800">
                    {safeRender(() => model.likes?.toLocaleString() || '0')}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Tags Section */}
          {model.tags && Array.isArray(model.tags) && model.tags.length > 0 && (
            <div className="mb-6">
              <h3 className="text-md font-semibold text-gray-700 mb-2">标签</h3>
              <div className="flex flex-wrap gap-2">
                {model.tags.map((tag, index) => (
                  <span key={index} className="bg-indigo-100 text-indigo-800 text-xs font-medium px-3 py-1 rounded-full">
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Debug Info Section (Toggleable) */}
          {import.meta.env.DEV && (
            <div className="mt-8 border-t pt-4">
              <button
                onClick={() => setShowDebugInfo(!showDebugInfo)}
                className="text-sm text-blue-600 hover:text-blue-800 mb-2"
              >
                {showDebugInfo ? '隐藏' : '显示'}调试信息
              </button>
              {showDebugInfo && (
                <div className="p-4 bg-gray-100 rounded-lg overflow-x-auto">
                  <pre className="text-xs">{JSON.stringify(model, null, 2)}</pre>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  } catch (renderError) {
    console.error('Fatal render error:', renderError);
    return (
      <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
        <strong className="font-bold">渲染错误: </strong>
        <span className="block sm:inline">渲染模型详情页面时发生错误。</span>
        <div className="mt-4 p-2 bg-white rounded overflow-x-auto">
          <pre className="text-xs">{JSON.stringify({ error: String(renderError), modelId, modelKeys: model ? Object.keys(model) : [] }, null, 2)}</pre>
        </div>
      </div>
    );
  }
};

export default ModelDetailPage; 