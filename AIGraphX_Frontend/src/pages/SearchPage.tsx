import React, { useState, useMemo } from 'react';
import { useDebounce } from '../hooks/useDebounce';

// 导入复用组件和 Hook
import SearchBar from '../components/search/SearchBar';
import SearchModeToggle, { SearchMode } from '../components/search/SearchModeToggle';
import TargetSelector, { SearchTarget } from '../components/search/TargetSelector';
import SortDropdown, { SortOption } from '../components/search/SortDropdown';
import ResultList, { ResultItem } from '../components/search/ResultList';
import FilterPanel, { Filters } from '../components/search/FilterPanel';
import Pagination from '../components/common/Pagination';
import { usePaperSearch, useModelSearch, PaperSearchParams, ModelSearchParams } from '../api/apiQueries';
// 导入 API 类型
import type { paths, components } from '../types/api';

// 类型别名
type PaperSearchSortBy = paths['/api/v1/search/papers/']['get']['parameters']['query']['sort_by'];
type ModelSearchSortBy = paths['/api/v1/search/models/']['get']['parameters']['query']['sort_by'];
// 使用 components.schemas 提取具体类型
type PaperResultItemSchema = components['schemas']['SearchResultItem'];
type ModelResultItemSchema = components['schemas']['HFSearchResultItem'];

// 常量
const ITEMS_PER_PAGE = 10;

// 定义排序选项
const paperSortOptions: SortOption[] = [
  { value: 'score', label: '相关性' },
  { value: 'published_date_desc', label: '最新' },
  { value: 'published_date_asc', label: '最早' },
];
const modelSortOptions: SortOption[] = [
    { value: 'score', label: '相关性' },
    { value: 'likes_desc', label: '最多赞' },
    { value: 'downloads_desc', label: '最多下载' },
    { value: 'last_modified_desc', label: '最近更新' },
];

const SearchPage: React.FC = () => {
  // --- 搜索条件状态 ---
  const [rawQuery, setRawQuery] = useState<string>('');
  const [searchMode, setSearchMode] = useState<SearchMode>('keyword');
  const [searchTarget, setSearchTarget] = useState<SearchTarget>('papers');
  const [sortBy, setSortBy] = useState<string>('score');
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [filters, setFilters] = useState<Filters>({});

  // --- Debounced 查询 ---
  const debouncedQuery = useDebounce(rawQuery, 500);

  // --- 动态排序选项 ---
  const currentSortOptions = useMemo(() => {
    return searchTarget === 'papers' ? paperSortOptions : modelSortOptions;
  }, [searchTarget]);

  // --- 确保 sortBy 在切换 target 后有效 ---
  React.useEffect(() => {
      const defaultSort = 'score';
      if (!currentSortOptions.some(opt => opt.value === sortBy)) {
          setSortBy(defaultSort);
      }
  }, [searchTarget, currentSortOptions, sortBy]);

  // --- API 查询参数准备 (包含 filters) ---
  const queryParams = useMemo(() => {
    let apiSortBy: PaperSearchSortBy | ModelSearchSortBy | null = null; // 允许 null
    let apiSortOrder: 'asc' | 'desc' = 'desc';
    let apiSearchType = searchMode; // 默认使用当前模式

    // --- 调整模型搜索的 search_type --- //
    if (searchTarget === 'models' && apiSearchType === 'hybrid') {
        apiSearchType = 'semantic'; // 模型不支持 hybrid，强制改为 semantic
        // 或者可以改为 'keyword'，取决于您的偏好
        // apiSearchType = 'keyword'; 
    }

    if (sortBy.includes('_')) {
      const parts = sortBy.split('_');
      apiSortOrder = parts.pop() as 'asc' | 'desc';
      const potentialSortBy = parts.join('_');
      if (searchTarget === 'papers' && ['score', 'published_date'].includes(potentialSortBy)) {
          apiSortBy = potentialSortBy as PaperSearchSortBy;
      } else if (searchTarget === 'models' && ['score', 'likes', 'downloads', 'last_modified'].includes(potentialSortBy)) {
          apiSortBy = potentialSortBy as ModelSearchSortBy;
      } else {
          apiSortBy = null; // 如果排序键不匹配当前目标，则重置
      }
    } else if (sortBy === 'score') {
      apiSortBy = 'score';
      apiSortOrder = 'desc';
    } else {
        apiSortBy = null; // 处理 sortBy 不是 score 且不含 _ 的情况
    }

    // 如果 sortBy 无效或与 target 不匹配，设置默认排序
    if (!apiSortBy) {
        if (searchTarget === 'papers') {
             apiSortBy = apiSearchType === 'keyword' ? 'published_date' : 'score';
        } else { // searchTarget === 'models'
             apiSortBy = apiSearchType === 'keyword' ? 'last_modified' : 'score';
        }
        apiSortOrder = 'desc';
    }

    // 合并基础查询参数和过滤器
    const baseParams = {
      query: debouncedQuery,
      search_type: apiSearchType, // 使用调整后的 search_type
      sort_by: apiSortBy,
      sort_order: apiSortOrder,
      skip: (currentPage - 1) * ITEMS_PER_PAGE,
      limit: ITEMS_PER_PAGE,
    };

    // 合并过滤器，确保类型正确，移除空值
    const currentFilters: Record<string, any> = {};
    // 恢复日期格式为 YYYY-MM-DD
    if (filters.year_from) currentFilters.date_from = `${filters.year_from}-01-01`; 
    if (filters.year_to) currentFilters.date_to = `${filters.year_to}-12-31`; 
    // 添加 pipeline_tag 过滤器
    if (searchTarget === 'models' && filters.pipeline_tag) {
        currentFilters.pipeline_tag = filters.pipeline_tag;
    }

    return { ...baseParams, ...currentFilters } as PaperSearchParams | ModelSearchParams;
  }, [debouncedQuery, searchMode, sortBy, currentPage, searchTarget, filters]);

  // --- 条件调用 React Query Hooks (参数已包含 filters) ---
  const paperSearchQuery = usePaperSearch(
      queryParams as PaperSearchParams,
      searchTarget === 'papers' && !!debouncedQuery
  );
  const modelSearchQuery = useModelSearch(
      queryParams as ModelSearchParams,
      searchTarget === 'models' && !!debouncedQuery
  );

  // --- 合并 Hook 结果 ---
  const { data, isLoading, isFetching, isError, error } = 
      searchTarget === 'papers' ? paperSearchQuery : modelSearchQuery;

  const searchResult = useMemo(() => {
    if (!data) return undefined;
    if (searchTarget === 'papers' && 'items' in data && Array.isArray(data.items) && (data.items.length === 0 || 'pwc_id' in data.items[0])) {
        return data as paths['/api/v1/search/papers/']['get']['responses']['200']['content']['application/json'];
    } else if (searchTarget === 'models' && 'items' in data && Array.isArray(data.items) && (data.items.length === 0 || 'model_id' in data.items[0])) {
        return data as paths['/api/v1/search/models/']['get']['responses']['200']['content']['application/json'];
    }
    return undefined;
  }, [data, searchTarget]);

  // --- 处理搜索提交 ---
  const handleSearch = (newQuery: string) => {
    setRawQuery(newQuery);
    setCurrentPage(1);
  };

  // --- 处理目标改变 ---
  const handleTargetChange = (newTarget: SearchTarget) => {
      setSearchTarget(newTarget);
      setCurrentPage(1);
      const defaultSort = 'score'; 
      setSortBy(defaultSort);
      setFilters({}); // 清空过滤器
  };

  // --- 处理排序改变 ---
  const handleSortChange = (newSortOption: string) => {
      setSortBy(newSortOption);
      setCurrentPage(1);
  };

  // --- 处理过滤改变 ---
  const handleFilterChange = (newFilters: Filters) => {
    setFilters(newFilters);
    setCurrentPage(1); // 过滤器变化时重置到第一页
  };

  // --- 处理分页改变 ---
  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };

  // --- 转换 API 结果为 ResultItem 格式 (根据 target) ---
  const results: ResultItem[] = useMemo(() => {
    if (!searchResult?.items) return [];

    if (searchTarget === 'papers') {
      return (searchResult.items as PaperResultItemSchema[]).map((item) => ({
        id: item.pwc_id,
        title: item.title ?? '未知标题',
        description: item.summary ?? undefined,
        type: 'paper',
        publishedDate: item.published_date ?? undefined,
        authors: item.authors ?? undefined,
        score: item.score,
      }));
    } else if (searchTarget === 'models') {
      return (searchResult.items as ModelResultItemSchema[]).map((item) => ({
        id: item.model_id,
        title: item.model_id ?? '未知模型 ID',
        description: item.pipeline_tag ?? undefined,
        type: 'model',
        tags: item.tags ?? [],
        likes: item.likes ?? 0,
        downloads: item.downloads ?? 0,
        lastModified: item.last_modified ?? undefined,
        score: item.score,
        author: item.author ?? undefined,
      }));
    }
    return [];
  }, [searchResult, searchTarget]);

  // --- 计算总页数 ---
  const totalPages = useMemo(() => {
    const totalItems = searchResult?.total ?? 0;
    return totalItems ? Math.ceil(totalItems / ITEMS_PER_PAGE) : 0;
  }, [searchResult]);

  // --- 页面标题 ---
  const pageTitle = searchTarget === 'papers' ? '搜索 AI 论文' : '搜索 AI 模型';

  return (
    <div className="space-y-8">
      {/* 上方控制区域 */}
      <div>
          <h2 className="text-xl font-semibold mb-6">{pageTitle}</h2>
          <div className="mb-4 p-4 bg-white rounded-md shadow-sm border border-gray-200">
            <SearchBar
              initialQuery={rawQuery}
              onSearch={handleSearch}
              isLoading={isFetching}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            {/* 左侧控制项 */}
            <div className="md:col-span-2 flex flex-wrap gap-x-4 gap-y-4 items-start">
              <SearchModeToggle searchMode={searchMode} onChange={setSearchMode} />
              <TargetSelector searchTarget={searchTarget} onChange={handleTargetChange} />
              <SortDropdown
                options={currentSortOptions}
                selectedOption={sortBy}
                onChange={handleSortChange}
              />
              {/* 过滤面板现在放在这里 */}
              <div className="w-full md:w-auto"> {/* 允许过滤面板占用更多空间 */}
                <FilterPanel
                  initialFilters={filters}
                  onFilterChange={handleFilterChange}
                  target={searchTarget === 'both' ? 'papers' : searchTarget}
                />
              </div>
            </div>

            {/* 空白占位符或右侧其他内容（如果需要）*/}
            {/* <div className="md:col-span-1"></div> */}
          </div>
      </div>


      {/* 下方结果区域 */}
      <div>
          {isError && (
            <div className="text-red-600 bg-red-100 p-4 rounded-md">
              搜索出错: {error instanceof Error ? error.message : '未知错误'}
            </div>
          )}

          {isLoading ? (
            <div className="text-center py-10">加载中...</div>
          ) : searchResult && searchResult.items && searchResult.items.length > 0 ? (
            <>
              {/* 传递 target, isLoading, query 给 ResultList */}
              <ResultList 
                results={results} 
                target={searchTarget === 'both' ? 'papers' : searchTarget} 
                isLoading={isLoading} 
                query={debouncedQuery}
              />
              {totalPages > 1 && (
                <Pagination
                  currentPage={currentPage}
                  totalPages={totalPages}
                  onPageChange={handlePageChange}
                />
              )}
            </>
          ) : debouncedQuery ? (
             <div className="text-center py-10 text-gray-500">找不到结果。</div>
          ) : (
             <div className="text-center py-10 text-gray-500">请输入关键词开始搜索。</div>
          ) }
      </div>
    </div>
  );
};

export default SearchPage; 