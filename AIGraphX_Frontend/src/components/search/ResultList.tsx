import React from 'react';
import { Link } from 'react-router-dom';
import type { SearchTarget } from './TargetSelector'; // 引入 SearchTarget 类型

// 搜索结果类型
export type ResultItemType = 'paper' | 'model';

// 搜索结果项接口
export interface ResultItem {
  id: string;
  title: string;
  description?: string;
  type: ResultItemType;
  publishedDate?: string;
  authors?: string[];
  venue?: string;
  arxivId?: string;
  downloadCount?: number;
  starCount?: number;
  task?: string;
  // 从 SearchPage 传过来的新字段
  tags?: string[];
  likes?: number;
  downloads?: number;
  lastModified?: string;
  score?: number | null;
  author?: string;
}

interface ResultListProps {
  results: ResultItem[];
  isLoading: boolean;
  query: string;
  target: SearchTarget; // 添加 target 属性
}

const ResultList: React.FC<ResultListProps> = ({ results, isLoading, query, target }) => {
  if (isLoading) {
    return (
      <div className="flex justify-center py-8">
        <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (results.length === 0) {
    if (query) {
      return (
        <div className="text-center py-8 text-gray-500">
          没有找到{target === 'papers' ? '论文' : '模型'}结果，请尝试其他关键词
        </div>
      );
    }
    return (
      <div className="text-center py-8 text-gray-500">
        输入关键词开始搜索{target === 'papers' ? '论文' : '模型'}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {results.map((result) => (
        <div 
          key={`${result.type}-${result.id}`}
          className="bg-white p-4 rounded-md shadow hover:shadow-md transition-shadow"
        >
          <Link 
            to={result.type === 'paper' ? `/papers/${result.id}` : `/models/${result.id}`}
            className="block"
          >
            <h3 className="text-lg font-medium text-blue-700 hover:text-blue-900">{result.title}</h3>
            
            {result.description && (
              <p className="mt-1 text-gray-600 line-clamp-2">{result.description}</p>
            )}
            
            <div className="mt-2 flex flex-wrap gap-x-4 text-sm text-gray-500">
              {result.type === 'paper' ? (
                <>
                  <span className="inline-flex items-center" title={result.authors?.join(', ')}>
                    <svg className="mr-1 h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                    {result.authors?.slice(0, 3).join(', ')}
                    {result.authors && result.authors.length > 3 && ' 等'}
                  </span>
                  
                  {result.publishedDate && (
                    <span className="inline-flex items-center">
                      <svg className="mr-1 h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                      {result.publishedDate}
                    </span>
                  )}
                  
                  {result.score !== undefined && result.score !== null && (
                     <span className="inline-flex items-center" title={`相关性分数: ${result.score.toFixed(3)}`}>
                       <svg className="mr-1 h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                         <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                       </svg>
                       分数: {result.score.toFixed(2)}
                     </span>
                  )}
                </>
              ) : (
                <>
                  {result.author && (
                    <span className="inline-flex items-center">
                      <svg className="mr-1 h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                      </svg>
                      {result.author}
                    </span>
                  )}
                  {result.likes !== undefined && (
                    <span className="inline-flex items-center">
                      <svg className="mr-1 h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                      </svg>
                      {result.likes.toLocaleString()} 赞
                    </span>
                  )}
                  {result.downloads !== undefined && (
                    <span className="inline-flex items-center">
                      <svg className="mr-1 h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                      </svg>
                      {result.downloads.toLocaleString()} 下载
                    </span>
                  )}
                  {result.score !== undefined && result.score !== null && (
                     <span className="inline-flex items-center" title={`相关性分数: ${result.score.toFixed(3)}`}>
                       <svg className="mr-1 h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                         <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                       </svg>
                       分数: {result.score.toFixed(2)}
                     </span>
                   )}
                  {result.lastModified && (
                     <span className="inline-flex items-center">
                       <svg className="mr-1 h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                         <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                       </svg>
                       {new Date(result.lastModified).toLocaleDateString()}
                     </span>
                   )}
                </>
              )}
              
              <span className="inline-flex items-center bg-gray-100 px-2 py-0.5 rounded-full text-xs font-medium text-gray-800">
                {result.type === 'paper' ? '论文' : '模型'}
              </span>
            </div>
          </Link>
        </div>
      ))}
    </div>
  );
};

export default ResultList; 