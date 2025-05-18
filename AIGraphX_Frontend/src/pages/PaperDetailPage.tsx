import React, { useState } from 'react';
import { useParams } from 'react-router-dom';
import { usePaperDetail } from '../api/apiQueries';
import Spinner from '../components/common/Spinner';
import { DocumentTextIcon, LinkIcon, TagIcon, StarIcon, BeakerIcon, CircleStackIcon, UserGroupIcon } from '@heroicons/react/24/outline';

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
        <p className="ml-2 text-gray-600">正在加载论文详情...</p>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
        <strong className="font-bold">获取论文详情出错: </strong>
        <span className="block sm:inline">{error instanceof Error ? error.message : JSON.stringify(error)}</span>
        <p className="text-sm mt-1">请求的 PWC ID: {pwcId}</p>
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
    <div className="bg-white rounded-lg shadow-lg overflow-hidden">
      <div className="p-6 bg-gradient-to-r from-blue-50 to-purple-50 border-b border-gray-200">
        <h1 className="text-2xl md:text-3xl font-bold text-gray-900 mb-2">{paper.title ?? '未知标题'}</h1>
        <div className="flex flex-wrap text-sm text-gray-600 gap-x-4 gap-y-1">
          {paper.published_date && (
            <span className="inline-flex items-center">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              发表于 {new Date(paper.published_date).toLocaleDateString('zh-CN', { year: 'numeric', month: 'long', day: 'numeric' })}
            </span>
          )}
          {paper.area && (
            <span className="inline-flex items-center">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1.5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                 <path strokeLinecap="round" strokeLinejoin="round" d="M5 15l7-7 7 7" />
              </svg>
              领域: <span className="font-medium text-indigo-700 ml-1">{paper.area}</span>
            </span>
          )}
          {paper.conference && (
            <span className="inline-flex items-center">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="h-4 w-4 mr-1.5 text-gray-400">
                <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 21h16.5M4.5 3h15M5.25 3v18m13.5-18v18M9 6.75h6M9 11.25h6M9 15.75h6M5.25 21v-3.375c0-.621.504-1.125 1.125-1.125h11.25c.621 0 1.125.504 1.125 1.125V21" />
              </svg>
              会议: <span className="font-medium text-teal-700 ml-1">{paper.conference}</span>
            </span>
          )}
        </div>
      </div>

      <div className="p-6 space-y-6">
        {paper.authors && paper.authors.length > 0 && (
          <div>
            <h3 className="text-md font-semibold text-gray-800 mb-2 inline-flex items-center">
               <UserGroupIcon className="h-5 w-5 mr-2 text-gray-400"/> 作者
            </h3>
            <p className="text-sm text-gray-700 leading-relaxed">
              {paper.authors.join(', ')}
            </p>
          </div>
        )}

        {paper.abstract && (
          <div className="border-t border-gray-200 pt-6">
            <h2 className="text-lg font-semibold mb-2 text-gray-800 inline-flex items-center">
              <DocumentTextIcon className="h-5 w-5 mr-2 text-gray-500"/> 摘要
            </h2>
            <div className="prose prose-sm max-w-none text-gray-700 leading-relaxed">
                 <p>{paper.abstract}</p> 
            </div>
          </div>
        )}

        <div className="border-t border-gray-200 pt-6 flex flex-wrap items-center gap-3">
          <h3 className="text-md font-semibold text-gray-800 mr-2">资源:</h3>
          {paper.url_abs && (
             <a href={paper.url_abs} target="_blank" rel="noopener noreferrer" className="inline-flex items-center px-3 py-1 bg-gray-100 text-gray-800 rounded-full text-sm font-medium hover:bg-gray-200 transition shadow-sm">
               <LinkIcon className="w-4 h-4 mr-1.5"/> 原文
             </a>
           )}
          {paper.url_pdf && (
            <a href={paper.url_pdf} target="_blank" rel="noopener noreferrer" className="inline-flex items-center px-3 py-1 bg-red-100 text-red-800 rounded-full text-sm font-medium hover:bg-red-200 transition shadow-sm">
               <svg className="w-4 h-4 mr-1.5" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M6 2a2 2 0 00-2 2v12a2 2 0 002 2h8a2 2 0 002-2V7.414A2 2 0 0015.414 6L12 2.586A2 2 0 0010.586 2H6zm2 10a1 1 0 10-2 0v3a1 1 0 102 0v-3zm2-3a1 1 0 011 1v5a1 1 0 11-2 0v-5a1 1 0 011-1zm4-1a1 1 0 10-2 0v7a1 1 0 102 0V8z" clipRule="evenodd"></path></svg>
               PDF
            </a>
          )}
          {paper.arxiv_id && (
            <span className="inline-flex items-center px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium shadow-sm">
               <TagIcon className="w-4 h-4 mr-1.5"/> arXiv: {paper.arxiv_id}
             </span>
           )}
        </div>

        {( (paper.tasks && paper.tasks.length > 0) || 
           (paper.methods && paper.methods.length > 0) || 
           (paper.datasets && paper.datasets.length > 0) 
         ) && (
          <div className="border-t border-gray-200 pt-6 grid grid-cols-1 md:grid-cols-3 gap-x-6 gap-y-4 text-sm">
            {paper.tasks && paper.tasks.length > 0 && (
              <div>
                   <h3 className="font-semibold text-gray-800 mb-2 inline-flex items-center"><TagIcon className="h-4 w-4 mr-1.5 text-indigo-500"/>相关任务</h3>
                   <div className="flex flex-wrap gap-1.5">
                       {paper.tasks?.map(task => (
                          <span key={task} className="bg-indigo-100 text-indigo-800 text-xs font-semibold px-2.5 py-0.5 rounded-md shadow-sm">
                              {task}
                          </span>
                       ))}
                   </div>
              </div>
            )}
            {paper.methods && paper.methods.length > 0 && (
               <div>
                    <h3 className="font-semibold text-gray-800 mb-2 inline-flex items-center"><BeakerIcon className="h-4 w-4 mr-1.5 text-purple-500"/>使用方法</h3>
                    <div className="flex flex-wrap gap-1.5">
                        {paper.methods?.map(method => (
                           <span key={method} className="bg-purple-100 text-purple-800 text-xs font-semibold px-2.5 py-0.5 rounded-md shadow-sm">
                               {method}
                           </span>
                        ))}
                    </div>
               </div>
             )}
             {paper.datasets && paper.datasets.length > 0 && (
                <div>
                     <h3 className="font-semibold text-gray-800 mb-2 inline-flex items-center"><CircleStackIcon className="h-4 w-4 mr-1.5 text-pink-500"/>使用数据集</h3>
                     <div className="flex flex-wrap gap-1.5">
                         {paper.datasets?.map(dataset => (
                            <span key={dataset} className="bg-pink-100 text-pink-800 text-xs font-semibold px-2.5 py-0.5 rounded-md shadow-sm">
                                {dataset}
                            </span>
                         ))}
                     </div>
                </div>
              )}
          </div>
        )}
      </div>
    </div>
  );
};

export default PaperDetailPage; 