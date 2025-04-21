import React from 'react';
import { useParams, Link } from 'react-router-dom';
import { usePaperDetail } from '../api/apiQueries';
import Spinner from '../components/common/Spinner';

// 论文详情类型接口（临时示例）
interface PaperDetail {
  id: string;
  title: string;
  abstract: string;
  authors: string[];
  publishedDate: string;
  venue: string;
  arxivId: string;
  doi?: string;
  citations: number;
  relatedModels: Array<{
    id: string;
    name: string;
  }>;
  codeLinks: Array<{
    url: string;
    provider: string;
  }>;
}

const PaperDetailPage: React.FC = () => {
  const { pwcId } = useParams<{ pwcId: string }>();

  const {
    data: paper,
    isLoading,
    isError,
    error,
  } = usePaperDetail(pwcId);

  if (isLoading) {
    return (
      <div className="flex justify-center items-center py-20">
        <Spinner />
      </div>
    );
  }

  if (isError) {
    return (
      <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
        <strong className="font-bold">获取论文详情出错: </strong>
        <span className="block sm:inline">{error instanceof Error ? error.message : JSON.stringify(error)}</span>
      </div>
    );
  }

  if (!paper) {
    return (
      <div className="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded relative" role="alert">
        <strong className="font-bold">未找到: </strong>
        <span className="block sm:inline">无法找到 PWC ID 为 {pwcId} 的论文。</span>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h1 className="text-2xl font-bold text-gray-800 mb-4">{paper.title ?? '未知标题'}</h1>

      <div className="mb-6 text-sm text-gray-600">
        {/* 作者 */} 
        {paper.authors && paper.authors.length > 0 && (
          <p className="mb-2">
            <span className="font-medium text-gray-800">作者: </span>
            {paper.authors.join(', ')}
          </p>
        )}
        {/* 发表日期 */} 
        {paper.published_date && (
           <p className="mb-2">
               <span className="font-medium text-gray-800">发表日期: </span>
               {new Date(paper.published_date).toLocaleDateString()}
           </p>
        )}
        {/* 领域 */} 
        {paper.area && (
            <p className="mb-2">
                <span className="font-medium text-gray-800">领域: </span>
                {paper.area}
            </p>
        )}
      </div>


      {/* 链接和徽章 */} 
      <div className="flex flex-wrap gap-2 mb-6">
          {paper.url_abs && (
             <a
               href={paper.url_abs}
               target="_blank"
               rel="noopener noreferrer"
               className="inline-flex items-center px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium hover:bg-blue-200"
             >
               <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20"><path d="M11 3a1 1 0 100 2h2.586l-6.293 6.293a1 1 0 101.414 1.414L15 6.414V9a1 1 0 102 0V4a1 1 0 00-1-1h-5z"></path><path d="M5 5a2 2 0 00-2 2v8a2 2 0 002 2h8a2 2 0 002-2v-3a1 1 0 10-2 0v3H5V7h3a1 1 0 000-2H5z"></path></svg>
               原文链接
             </a>
           )}
          {paper.url_pdf && (
            <a
              href={paper.url_pdf}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center px-3 py-1 bg-red-100 text-red-800 rounded-full text-sm font-medium hover:bg-red-200"
            >
               <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M6 2a2 2 0 00-2 2v12a2 2 0 002 2h8a2 2 0 002-2V7.414A2 2 0 0015.414 6L12 2.586A2 2 0 0010.586 2H6zm2 10a1 1 0 10-2 0v3a1 1 0 102 0v-3zm2-3a1 1 0 011 1v5a1 1 0 11-2 0v-5a1 1 0 011-1zm4-1a1 1 0 10-2 0v7a1 1 0 102 0V8z" clipRule="evenodd"></path></svg>
              PDF
            </a>
          )}
          {paper.arxiv_id && (
            <span className="inline-flex items-center px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
               <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7A2 2 0 0112 21H7a2 2 0 01-2-2V5a2 2 0 012-2z"></path></svg>
              arXiv: {paper.arxiv_id}
            </span>
          )}
          {paper.number_of_stars !== undefined && paper.number_of_stars !== null && (
             <span className="inline-flex items-center px-3 py-1 bg-yellow-100 text-yellow-800 rounded-full text-sm font-medium">
               <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"></path></svg>
               {paper.number_of_stars} Stars
             </span>
           )}
      </div>

      {/* 摘要 */} 
      {paper.abstract && (
        <div className="mb-8">
          <h2 className="text-lg font-semibold mb-2 text-gray-800">摘要</h2>
          <p className="text-gray-700 leading-relaxed">{paper.abstract}</p>
        </div>
      )}

      {/* 任务、方法、数据集 */} 
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8 text-sm">
          {paper.tasks && paper.tasks.length > 0 && (
            <div>
                 <h3 className="font-medium text-gray-800 mb-2">相关任务</h3>
                 <div className="flex flex-wrap gap-1">
                     {paper.tasks.map(task => (
                        <span key={task} className="bg-indigo-100 text-indigo-800 text-xs font-medium px-2.5 py-0.5 rounded">
                            {task}
                        </span>
                     ))}
                 </div>
            </div>
          )}
          {paper.methods && paper.methods.length > 0 && (
             <div>
                  <h3 className="font-medium text-gray-800 mb-2">使用方法</h3>
                  <div className="flex flex-wrap gap-1">
                      {paper.methods.map(method => (
                         <span key={method} className="bg-purple-100 text-purple-800 text-xs font-medium px-2.5 py-0.5 rounded">
                             {method}
                         </span>
                      ))}
                  </div>
             </div>
           )}
           {paper.datasets && paper.datasets.length > 0 && (
              <div>
                   <h3 className="font-medium text-gray-800 mb-2">使用数据集</h3>
                   <div className="flex flex-wrap gap-1">
                       {paper.datasets.map(dataset => (
                          <span key={dataset} className="bg-pink-100 text-pink-800 text-xs font-medium px-2.5 py-0.5 rounded">
                              {dataset}
                          </span>
                       ))}
                   </div>
              </div>
            )}
      </div>


      {/* TODO: 添加显示相关模型或其他实体的部分 */} 
      {/* <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div>
          <h2 className="text-xl font-semibold mb-4">相关模型</h2>
          <p className="text-gray-500">相关模型信息待实现</p>
        </div>
        <div>
            <h2 className="text-xl font-semibold mb-4">代码链接</h2>
            <p className="text-gray-500">代码链接信息待实现</p>
        </div>
      </div> */}
    </div>
  );
};

export default PaperDetailPage; 