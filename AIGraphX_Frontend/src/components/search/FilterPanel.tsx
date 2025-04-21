import React, { useState, useEffect } from 'react';

// 定义过滤参数类型 (根据需要扩展)
export interface Filters {
  year_from?: number | null;
  year_to?: number | null;
  pipeline_tag?: string | null; // 添加 pipeline_tag 过滤器
  // 可以添加其他过滤，例如:
  // area?: string | null;
}

interface FilterPanelProps {
  initialFilters: Filters;
  onFilterChange: (newFilters: Filters) => void;
  target: 'papers' | 'models'; // 根据目标显示不同的过滤器
}

// 预定义的常见 pipeline tags (可以从 API 获取或扩展)
const commonPipelineTags = [
  { value: 'text-generation', label: '文本生成' },
  { value: 'fill-mask', label: '填充掩码' },
  { value: 'token-classification', label: '令牌分类' },
  { value: 'sentence-similarity', label: '句子相似度' },
  { value: 'text-classification', label: '文本分类' },
  { value: 'question-answering', label: '问答' },
  { value: 'summarization', label: '摘要' },
  { value: 'translation', label: '翻译' },
  { value: 'feature-extraction', label: '特征提取' },
  { value: 'text-to-image', label: '文本到图像' },
  { value: 'image-to-text', label: '图像到文本' },
  { value: 'image-classification', label: '图像分类' },
  { value: 'object-detection', label: '目标检测' },
  { value: 'audio-classification', label: '音频分类' },
  { value: 'automatic-speech-recognition', label: '语音识别' },
];

const FilterPanel: React.FC<FilterPanelProps> = ({ initialFilters, onFilterChange, target }) => {
  // 使用状态来管理过滤器的当前值
  const [yearFrom, setYearFrom] = useState<string>('');
  const [yearTo, setYearTo] = useState<string>('');
  const [selectedPipelineTag, setSelectedPipelineTag] = useState<string>('');

  // 当 initialFilters 改变时，更新本地状态
  useEffect(() => {
    setYearFrom(initialFilters.year_from?.toString() ?? '');
    setYearTo(initialFilters.year_to?.toString() ?? '');
    setSelectedPipelineTag(initialFilters.pipeline_tag ?? '');
  }, [initialFilters]);

  const handleApplyFilters = () => {
    const newFilters: Filters = {
      year_from: yearFrom ? parseInt(yearFrom, 10) : null,
      year_to: yearTo ? parseInt(yearTo, 10) : null,
      pipeline_tag: selectedPipelineTag || null,
    };
    // 移除值为 null 或 NaN 或空字符串的键
    Object.keys(newFilters).forEach(key => {
      const filterKey = key as keyof Filters;
      const value = newFilters[filterKey];
      if (value === null || value === '' || (typeof value === 'number' && isNaN(value))) {
        delete newFilters[filterKey];
      }
    });
    onFilterChange(newFilters);
  };

  const handleClearFilters = () => {
      setYearFrom('');
      setYearTo('');
      setSelectedPipelineTag('');
      onFilterChange({}); // 清空所有过滤器
  };

  // 根据目标决定显示哪些过滤器
  const showYearFilter = target === 'papers';
  const showPipelineTagFilter = target === 'models';

  return (
    <div className="p-4 bg-white rounded-lg shadow-sm border border-gray-200">
      <h3 className="text-md font-semibold mb-3 text-gray-700">过滤选项</h3>
      <div className="space-y-4">
        
        {/* 年份范围过滤器 (仅论文) */}
        {showYearFilter && (
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-600">发表年份范围</label>
            <div className="flex items-center space-x-2">
              <input 
                type="number" 
                placeholder="从 (例如 2020)" 
                value={yearFrom}
                onChange={(e) => setYearFrom(e.target.value)}
                className="w-full px-2 py-1 border border-gray-300 rounded-md text-sm focus:ring-blue-500 focus:border-blue-500"
              />
              <span>-</span>
              <input 
                type="number" 
                placeholder="到 (例如 2023)" 
                value={yearTo}
                onChange={(e) => setYearTo(e.target.value)}
                className="w-full px-2 py-1 border border-gray-300 rounded-md text-sm focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
          </div>
        )}

        {/* Pipeline Tag 过滤器 (仅模型) */}
        {showPipelineTagFilter && (
            <div className="space-y-2">
                <label htmlFor="pipelineTagSelect" className="block text-sm font-medium text-gray-600">模型任务 (Pipeline Tag)</label>
                <select 
                    id="pipelineTagSelect"
                    value={selectedPipelineTag}
                    onChange={(e) => setSelectedPipelineTag(e.target.value)}
                    className="w-full px-2 py-1 border border-gray-300 rounded-md text-sm focus:ring-blue-500 focus:border-blue-500 bg-white"
                >
                    <option value="">所有任务</option>
                    {commonPipelineTags.map(tag => (
                        <option key={tag.value} value={tag.value}>
                            {tag.label}
                        </option>
                    ))}
                </select>
            </div>
        )}

      </div>

      {/* 操作按钮 */}
      <div className="mt-5 flex justify-end space-x-2">
          <button
              onClick={handleClearFilters}
              className="px-3 py-1 border border-gray-300 rounded-md text-sm text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          >
              清除
          </button>
          <button 
            onClick={handleApplyFilters} 
            className="px-3 py-1 bg-blue-600 text-white rounded-md text-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            应用
          </button>
      </div>
    </div>
  );
};

export default FilterPanel; 