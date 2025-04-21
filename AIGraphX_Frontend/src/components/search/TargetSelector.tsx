import React from 'react';

export type SearchTarget = 'papers' | 'models' | 'both';

interface TargetSelectorProps {
  searchTarget: SearchTarget;
  onChange: (target: SearchTarget) => void;
}

const TargetSelector: React.FC<TargetSelectorProps> = ({ searchTarget, onChange }) => {
  return (
    <div>
      <span className="mr-2 text-gray-700">搜索目标:</span>
      <div className="inline-flex rounded-md shadow-sm">
        <button
          type="button"
          className={`px-3 py-1 text-sm rounded-l-md ${
            searchTarget === 'papers' 
              ? 'bg-blue-600 text-white' 
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
          onClick={() => onChange('papers')}
        >
          论文
        </button>
        <button
          type="button"
          className={`px-3 py-1 text-sm ${
            searchTarget === 'models' 
              ? 'bg-blue-600 text-white' 
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
          onClick={() => onChange('models')}
        >
          模型
        </button>
        <button
          type="button"
          className={`px-3 py-1 text-sm rounded-r-md ${
            searchTarget === 'both' 
              ? 'bg-blue-600 text-white' 
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
          onClick={() => onChange('both')}
        >
          全部
        </button>
      </div>
    </div>
  );
};

export default TargetSelector; 