import React, { useState, useEffect } from 'react';
import { usePaperAreas } from '../../api/apiQueries'; // 更新导入路径

// 定义过滤参数类型 (根据需要扩展)
export interface Filters {
  year_from?: number | null;
  year_to?: number | null;
  pipeline_tag?: string | null; // 添加 pipeline_tag 过滤器
  area?: string[]; // 更新为字符串数组，支持多选
  library_name?: string | null; // 模型库
  tags?: string[]; // 模型标签 (数组形式)
  author?: string | null; // 模型作者
  paper_author?: string[] | null; // 论文作者 (改为字符串数组)
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
  const [selectedAreas, setSelectedAreas] = useState<string[]>([]); // 添加领域多选状态
  const [libraryName, setLibraryName] = useState<string>('');
  const [tagsInput, setTagsInput] = useState<string>('');
  const [authorInput, setAuthorInput] = useState<string>('');
  const [paperAuthorInput, setPaperAuthorInput] = useState<string>('');
  
  // 使用 React Query 获取可用的论文领域
  const { data: availableAreas = [], isLoading: isLoadingAreas } = usePaperAreas();

  // 当 initialFilters 改变时，更新本地状态
  useEffect(() => {
    setYearFrom(initialFilters.year_from?.toString() ?? '');
    setYearTo(initialFilters.year_to?.toString() ?? '');
    setSelectedPipelineTag(initialFilters.pipeline_tag ?? '');
    setSelectedAreas(initialFilters.area || []);
    setLibraryName(initialFilters.library_name ?? '');
    setTagsInput(initialFilters.tags?.join(', ') ?? '');
    setAuthorInput(initialFilters.author ?? '');
    setPaperAuthorInput(initialFilters.paper_author?.join(', ') ?? ''); // 将数组转为逗号分隔字符串用于输入框
  }, [initialFilters]);

  const handleApplyFilters = () => {
    const tagsArray = tagsInput.split(',').map(tag => tag.trim()).filter(tag => tag !== '');
    // 将输入的单个论文作者转为单元素数组（如果非空）
    const paperAuthorArray = paperAuthorInput.trim() ? [paperAuthorInput.trim()] : null;

    const newFilters: Filters = {
      year_from: yearFrom ? parseInt(yearFrom, 10) : null,
      year_to: yearTo ? parseInt(yearTo, 10) : null,
      pipeline_tag: selectedPipelineTag || null,
      area: selectedAreas.length > 0 ? selectedAreas : undefined, // 只在有选中值时添加
      library_name: libraryName.trim() || null,
      tags: tagsArray.length > 0 ? tagsArray : undefined,
      author: authorInput.trim() || null,
      paper_author: paperAuthorArray, // 使用处理后的数组
    };
    // 移除值为 null 或 NaN 或空字符串或空数组的键
    Object.keys(newFilters).forEach(key => {
      const filterKey = key as keyof Filters;
      const value = newFilters[filterKey];
      if (value === null || value === '' || 
          (typeof value === 'number' && isNaN(value)) || 
          (Array.isArray(value) && value.length === 0)) {
        delete newFilters[filterKey];
      }
    });
    onFilterChange(newFilters);
    // 不再需要在应用后清除输入框，因为状态会从 SearchPage 的 filters 状态更新
    // setSelectedPipelineTag('');
    // setSelectedAreas([]);
    // setLibraryName('');
    // setTagsInput('');
    // setAuthorInput('');
    // setPaperAuthorInput('');
  };

  const handleClearFilters = () => {
    setYearFrom('');
    setYearTo('');
    setSelectedPipelineTag('');
    setSelectedAreas([]);
    setLibraryName('');
    setTagsInput('');
    setAuthorInput('');
    setPaperAuthorInput('');
    onFilterChange({}); // 清空所有过滤器
  };

  // 处理领域选中状态变化
  const handleAreaToggle = (area: string) => {
    setSelectedAreas(prev => 
      prev.includes(area) 
        ? prev.filter(a => a !== area) // 如果已选中，则移除
        : [...prev, area] // 如果未选中，则添加
    );
  };

  // 根据目标决定显示哪些过滤器
  const showYearFilter = target === 'papers';
  const showPipelineTagFilter = target === 'models';
  const showAreaFilter = target === 'papers';
  const showLibraryFilter = target === 'models';
  const showTagsFilter = target === 'models';
  const showAuthorFilter = target === 'models';
  const showPaperAuthorFilter = target === 'papers';

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

        {/* 论文作者过滤器 (仅论文) */}
        {showPaperAuthorFilter && (
            <div className="space-y-2">
                <label htmlFor="paperAuthorInput" className="block text-sm font-medium text-gray-600">论文作者</label>
                <input
                    id="paperAuthorInput"
                    type="text"
                    placeholder="输入作者姓名"
                    value={paperAuthorInput}
                    onChange={(e) => setPaperAuthorInput(e.target.value)}
                    className="w-full px-2 py-1 border border-gray-300 rounded-md text-sm focus:ring-blue-500 focus:border-blue-500"
                />
            </div>
        )}

        {/* 论文领域多选过滤器 (仅论文) */}
        {showAreaFilter && (
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-600">研究领域</label>
            {isLoadingAreas ? (
              <div className="text-sm text-gray-500">加载中...</div>
            ) : availableAreas.length > 0 ? (
              <div className="max-h-40 overflow-y-auto p-1 border border-gray-200 rounded-md">
                {availableAreas.map(area => (
                  <div key={area} className="flex items-center mb-1">
                    <input
                      id={`area-${area}`}
                      type="checkbox"
                      checked={selectedAreas.includes(area)}
                      onChange={() => handleAreaToggle(area)}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    <label
                      htmlFor={`area-${area}`}
                      className="ml-2 block text-sm text-gray-700"
                    >
                      {area}
                    </label>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-sm text-gray-500">暂无可用领域</div>
            )}
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

        {/* 模型库过滤器 (仅模型) */}
        {showLibraryFilter && (
            <div className="space-y-2">
                <label htmlFor="libraryNameInput" className="block text-sm font-medium text-gray-600">模型库 (例如 transformers)</label>
                <input
                    id="libraryNameInput"
                    type="text"
                    placeholder="输入库名称"
                    value={libraryName}
                    onChange={(e) => setLibraryName(e.target.value)}
                    className="w-full px-2 py-1 border border-gray-300 rounded-md text-sm focus:ring-blue-500 focus:border-blue-500"
                />
            </div>
        )}

        {/* 模型标签过滤器 (仅模型, 逗号分隔) */}
        {showTagsFilter && (
            <div className="space-y-2">
                <label htmlFor="tagsInput" className="block text-sm font-medium text-gray-600">模型标签 (逗号分隔)</label>
                <input
                    id="tagsInput"
                    type="text"
                    placeholder="例如 vision, text-classification"
                    value={tagsInput}
                    onChange={(e) => setTagsInput(e.target.value)}
                    className="w-full px-2 py-1 border border-gray-300 rounded-md text-sm focus:ring-blue-500 focus:border-blue-500"
                />
            </div>
        )}

        {/* 模型作者过滤器 (仅模型) */}
        {showAuthorFilter && (
            <div className="space-y-2">
                <label htmlFor="authorInput" className="block text-sm font-medium text-gray-600">模型作者/机构</label>
                <input
                    id="authorInput"
                    type="text"
                    placeholder="输入作者或机构名称"
                    value={authorInput}
                    onChange={(e) => setAuthorInput(e.target.value)}
                    className="w-full px-2 py-1 border border-gray-300 rounded-md text-sm focus:ring-blue-500 focus:border-blue-500"
                />
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