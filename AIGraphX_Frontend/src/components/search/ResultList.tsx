import React from 'react';
import { Link } from 'react-router-dom';
import type { SearchTarget } from './TargetSelector'; // 引入 SearchTarget 类型
import { 
  UserGroupIcon, 
  CalendarDaysIcon, 
  SparklesIcon, 
  UserCircleIcon, 
  HeartIcon, 
  ArrowDownTrayIcon, 
  ClockIcon, 
  TagIcon 
} from '@heroicons/react/24/outline'; // 或者 /20/solid

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
  conference?: string;
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
  readmeContent?: string;
  datasetLinks?: string[];
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
          className="bg-white p-4 rounded-lg border border-gray-200 hover:shadow-lg transition-shadow duration-200 ease-in-out"
        >
          <Link 
            to={result.type === 'paper' ? `/papers/${result.id}` : `/models/${encodeURIComponent(result.id)}`}
            className="block"
          >
            <h3 className="text-lg font-semibold text-blue-700 hover:text-blue-900 mb-1.5 truncate">{result.title}</h3>
            
            {result.description && (
              <p className="mb-2.5 text-sm text-gray-600 line-clamp-2">{result.description}</p>
            )}
            
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-sm text-gray-500 mb-2.5">
              {result.type === 'paper' && (
                <>
                  {result.authors && result.authors.length > 0 && (
                    <span className="inline-flex items-center" title={result.authors?.join(', ')}>
                      <UserGroupIcon className="mr-1 h-4 w-4 text-gray-400" />
                      {result.authors?.slice(0, 2).join(', ')}
                      {result.authors.length > 2 && ' 等'}
                    </span>
                  )}
                  
                  {result.publishedDate && (
                    <span className="inline-flex items-center">
                      <CalendarDaysIcon className="mr-1 h-4 w-4 text-gray-400" />
                      {result.publishedDate}
                    </span>
                  )}
                  {result.conference && (
                    <span className="inline-flex items-center" title={`Conference: ${result.conference}`}>
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="mr-1 h-4 w-4 text-gray-400">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 7.5h1.5m-1.5 3h1.5m-7.5 3h7.5m-7.5 3h7.5m3-9h3.375c.621 0 1.125.504 1.125 1.125V18a2.25 2.25 0 01-2.25 2.25H6.75A2.25 2.25 0 014.5 18V7.875c0-.621.504-1.125 1.125-1.125H6.75m0-3h5.25m0 0h5.25M9 4.5V1.5M12 4.5V1.5m3 0V4.5m0 0V1.5" />
                      </svg>
                      {result.conference}
                    </span>
                  )}
                </>
              )}
              
              {result.type === 'model' && (
                <>
                  {result.author && (
                    <span className="inline-flex items-center">
                      <UserCircleIcon className="mr-1 h-4 w-4 text-gray-400" />
                      {result.author}
                    </span>
                  )}
                  {result.likes !== undefined && (
                    <span className="inline-flex items-center">
                      <HeartIcon className="mr-1 h-4 w-4 text-pink-500" />
                      {result.likes.toLocaleString()}
                    </span>
                  )}
                  {result.downloads !== undefined && (
                    <span className="inline-flex items-center">
                      <ArrowDownTrayIcon className="mr-1 h-4 w-4 text-green-500" />
                      {result.downloads.toLocaleString()}
                    </span>
                  )}
                  {result.lastModified && (
                    <span className="inline-flex items-center">
                      <ClockIcon className="mr-1 h-4 w-4 text-gray-400" />
                      {new Date(result.lastModified).toLocaleDateString()}
                    </span>
                  )}
                </>
              )}

              {result.score !== undefined && result.score !== null && (
                 <span className="inline-flex items-center text-purple-700" title={`相关性分数: ${result.score.toFixed(4)}`}>
                   <SparklesIcon className="mr-1 h-4 w-4 text-purple-500" />
                   {result.score.toFixed(2)}
                 </span>
              )}

            </div>

            {result.type === 'model' && result.tags && result.tags.length > 0 && (
              <div className="flex flex-wrap items-center gap-1.5 text-xs">
                <span className="font-medium text-gray-600"><TagIcon className="inline h-3.5 w-3.5 mr-0.5" />标签:</span>
                {result.tags.slice(0, 5).map(tag => (
                  <span key={tag} className="px-1.5 py-0.5 bg-blue-100 text-blue-800 rounded">{tag}</span>
                ))}
                {result.tags.length > 5 && (
                  <span className="text-gray-500">...</span>
                )}
              </div>
            )}

            {result.type === 'model' && result.readmeContent && (
              <div className="mt-2 pt-2 border-t border-gray-100">
                <h4 className="text-xs font-semibold text-gray-500 mb-0.5">README Snippet:</h4>
                <p className="text-xs text-gray-600 line-clamp-3 whitespace-pre-wrap">{result.readmeContent}</p>
              </div>
            )}

            {result.type === 'model' && result.datasetLinks && result.datasetLinks.length > 0 && (
              <div className="mt-2 pt-2 border-t border-gray-100">
                <h4 className="text-xs font-semibold text-gray-500 mb-0.5">Dataset Links:</h4>
                <ul className="list-disc list-inside pl-1 space-y-0.5">
                  {result.datasetLinks.slice(0, 3).map((link, index) => (
                    <li key={index} className="text-xs">
                      <a 
                        href={link}
                        target="_blank" 
                        rel="noopener noreferrer" 
                        className="text-blue-600 hover:text-blue-800 hover:underline truncate block"
                        title={link}
                      >
                        {link}
                      </a>
                    </li>
                  ))}
                  {result.datasetLinks.length > 3 && <li className="text-xs text-gray-500">...and {result.datasetLinks.length - 3} more</li>}
                </ul>
              </div>
            )}

          </Link>
        </div>
      ))}
    </div>
  );
};

export default ResultList; 