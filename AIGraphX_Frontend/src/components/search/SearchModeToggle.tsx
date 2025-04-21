import React from 'react';

export type SearchMode = 'keyword' | 'semantic' | 'hybrid';

interface SearchModeToggleProps {
  searchMode: SearchMode;
  onChange: (mode: SearchMode) => void;
}

const SearchModeToggle: React.FC<SearchModeToggleProps> = ({ searchMode, onChange }) => {
  return (
    <div>
      <span className="mr-2 text-gray-700">搜索模式:</span>
      <div className="inline-flex rounded-md shadow-sm">
        <button
          type="button"
          className={`px-3 py-1 text-sm rounded-l-md ${
            searchMode === 'keyword' 
              ? 'bg-blue-600 text-white' 
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
          onClick={() => onChange('keyword')}
        >
          关键词
        </button>
        <button
          type="button"
          className={`px-3 py-1 text-sm ${
            searchMode === 'semantic' 
              ? 'bg-blue-600 text-white' 
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
          onClick={() => onChange('semantic')}
        >
          语义
        </button>
        <button
          type="button"
          className={`px-3 py-1 text-sm rounded-r-md ${
            searchMode === 'hybrid' 
              ? 'bg-blue-600 text-white' 
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
          onClick={() => onChange('hybrid')}
        >
          混合
        </button>
      </div>
    </div>
  );
};

export default SearchModeToggle; 