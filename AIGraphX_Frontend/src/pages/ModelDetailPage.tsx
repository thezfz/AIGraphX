import React from 'react';
import { useParams } from 'react-router-dom';
import { useModelDetail } from '../api/apiQueries'; // 引入 Hook
import Spinner from '../components/common/Spinner'; // 引入 Spinner

const ModelDetailPage: React.FC = () => {
  const { modelId } = useParams<{ modelId: string }>();

  // 使用 React Query Hook 获取数据
  const {
    data: model, // API 返回的数据结构可能与之前的 ModelDetail 不同
    isLoading,
    isError,
    error,
  } = useModelDetail(modelId); // 传入 modelId

  if (isLoading) {
    return (
      <div className="flex justify-center items-center py-20">
        <Spinner /> {/* 使用 Spinner 组件 */}
      </div>
    );
  }

  if (isError) {
    return (
      <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
        <strong className="font-bold">获取模型详情出错: </strong>
        <span className="block sm:inline">{error instanceof Error ? error.message : JSON.stringify(error)}</span>
      </div>
    );
  }

  if (!model) {
    return (
      <div className="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded relative" role="alert">
        <strong className="font-bold">未找到: </strong>
        <span className="block sm:inline">无法找到 ID 为 {modelId} 的模型。</span>
      </div>
    );
  }

  // --- 渲染从 API 获取的模型数据 ---
  // 注意：这里的字段需要匹配 useModelDetail 返回的 ModelDetailResponse 类型
  // 例如，后端返回的可能是 HFModelDetail 结构
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h1 className="text-2xl font-bold text-gray-800 mb-4 break-all">{model.model_id}</h1> {/* 使用 model_id */}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="col-span-2">
          {/* 可以根据需要展示更多信息，比如从 README 获取描述 */}
          {/* <h2 className="text-lg font-semibold mb-2">描述</h2>
          <p className="text-gray-700 mb-4">{model.description}</p> */}

          <div className="grid grid-cols-2 gap-4 mb-4 text-sm">
             {model.pipeline_tag && (
                 <div>
                     <h3 className="text-md font-medium text-gray-600">任务类型</h3>
                     <p className="text-gray-800">{model.pipeline_tag}</p>
                 </div>
             )}
             {model.author && (
                <div>
                  <h3 className="text-md font-medium text-gray-600">作者/机构</h3>
                  <p className="text-gray-800">{model.author}</p>
                </div>
             )}
             {model.library_name && (
                <div>
                    <h3 className="text-md font-medium text-gray-600">库</h3>
                    <p className="text-gray-800">{model.library_name}</p>
                </div>
             )}
             {model.last_modified && (
                <div>
                  <h3 className="text-md font-medium text-gray-600">最后更新</h3>
                  <p className="text-gray-800">{new Date(model.last_modified).toLocaleDateString()}</p>
                </div>
             )}
             {/* 显示 Tags */} 
             {model.tags && model.tags.length > 0 && (
                <div className="col-span-2">
                     <h3 className="text-md font-medium text-gray-600 mb-1">标签</h3>
                     <div className="flex flex-wrap gap-2">
                         {model.tags.map(tag => (
                            <span key={tag} className="bg-blue-100 text-blue-800 text-xs font-medium px-2.5 py-0.5 rounded">
                                {tag}
                            </span>
                         ))}
                     </div>
                </div>
             )}
          </div>
        </div>

        <div className="bg-gray-50 rounded-lg p-4 self-start"> {/* 使用 self-start 避免拉伸 */}
          <h2 className="text-lg font-semibold mb-4">统计信息</h2>
          {model.downloads !== undefined && model.downloads !== null && (
            <div className="flex justify-between mb-2 text-sm">
              <span className="text-gray-600 inline-flex items-center">
                  <svg className="mr-1 h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>
                  下载次数:
              </span>
              <span className="font-medium text-gray-800">{model.downloads.toLocaleString()}</span>
            </div>
          )}
          {model.likes !== undefined && model.likes !== null && (
            <div className="flex justify-between text-sm">
              <span className="text-gray-600 inline-flex items-center">
                  <svg className="mr-1 h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" /></svg>
                  点赞数:
              </span>
              <span className="font-medium text-gray-800">{model.likes.toLocaleString()}</span>
            </div>
          )}
        </div>
      </div>

      {/* TODO: 添加显示相关论文或其他相关实体的部分 */}
      {/* <div className="mt-8">
        <h2 className="text-xl font-semibold mb-4">相关论文</h2>
        <p className="text-gray-500">相关论文信息待实现</p>
      </div> */}
    </div>
  );
};

export default ModelDetailPage; 