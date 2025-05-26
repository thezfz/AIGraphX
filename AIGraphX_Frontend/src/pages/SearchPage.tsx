import React, { useState, useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useDebounce } from '../hooks/useDebounce';
import { Link } from 'react-router-dom';
import { GlobeAltIcon } from '@heroicons/react/24/outline';

// 导入复用组件和 Hook
import SearchBar from '../components/search/SearchBar';
import SearchModeToggle, { SearchMode } from '../components/search/SearchModeToggle';
import TargetSelector, { SearchTarget as ActualSearchTarget } from '../components/search/TargetSelector';
import SortDropdown, { SortOption } from '../components/search/SortDropdown';
import ResultList, { ResultItem } from '../components/search/ResultList';
import FilterPanel, { Filters } from '../components/search/FilterPanel';
import Pagination from '../components/common/Pagination';
import { usePaperSearch, useModelSearch, PaperSearchParams, ModelSearchParams } from '../api/apiQueries';
// 导入 API 类型
import type { paths, components } from '../types/api';

// 类型别名 (移除 'both' 的可能性)
type SearchTarget = Exclude<ActualSearchTarget, 'both'>; // 显式排除 'both'
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
  const [searchParams, setSearchParams] = useSearchParams();

  // --- 从 URL 参数或默认值初始化状态 ---
  const [rawQuery, setRawQuery] = useState<string>(searchParams.get('q') || '');
  const [searchMode, setSearchMode] = useState<SearchMode>(searchParams.get('mode') as SearchMode || 'keyword');
  const [searchTarget, setSearchTarget] = useState<SearchTarget>(() => {
      const targetFromUrl = searchParams.get('target') as ActualSearchTarget;
      return targetFromUrl === 'papers' || targetFromUrl === 'models' ? targetFromUrl : 'papers';
  });
  const [sortBy, setSortBy] = useState<string>(searchParams.get('sort') || 'score');
  const [currentPage, setCurrentPage] = useState<number>(parseInt(searchParams.get('page') || '1', 10));
  // Filter 初始化稍微复杂，需要解析 URL 参数
  const [filters, setFilters] = useState<Filters>(() => {
    const initialFilters: Filters = {};
    // 示例：从 URL 读取日期过滤器 (需要确保 parse 和 stringify 对应)
    const dateFrom = searchParams.get('date_from');
    if (dateFrom) initialFilters.year_from = parseInt(dateFrom.split('-')[0], 10);
    const dateTo = searchParams.get('date_to');
    if (dateTo) initialFilters.year_to = parseInt(dateTo.split('-')[0], 10);
    // 示例：从 URL 读取 Area (假设用逗号分隔)
    const areaParam = searchParams.get('area');
    if (areaParam) initialFilters.area = areaParam.split(',');
    // 添加其他过滤器...
    initialFilters.pipeline_tag = searchParams.get('pipeline_tag') || undefined;
    initialFilters.library_name = searchParams.get('filter_library_name') || undefined;
    const tagsParam = searchParams.get('filter_tags');
    if (tagsParam) initialFilters.tags = tagsParam.split(',');
    initialFilters.author = searchParams.get('filter_author') || undefined;
    const paperAuthorsParam = searchParams.get('filter_authors');
    if (paperAuthorsParam) initialFilters.paper_author = paperAuthorsParam.split(',');

    return initialFilters;
  });

  // --- Debounced 查询 ---
  const debouncedQuery = useDebounce(rawQuery, 500);

  // --- 动态排序选项 ---
  const currentSortOptions = useMemo(() => {
    return searchTarget === 'papers' ? paperSortOptions : modelSortOptions;
  }, [searchTarget]);

  // --- 确保 sortBy 在切换 target 后有效 ---
  React.useEffect(() => {
      const defaultSort = 'score'; // 默认值可能需要根据 target 调整，但 score 对两者都有效
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
        // 删除此限制，因为后端现在支持模型的混合搜索
        // apiSearchType = 'semantic'; // 不再需要将 hybrid 强制转换为 semantic
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
    // 添加 area 过滤器 (修正：直接传递数组，因为后端支持 ANY)
    if (searchTarget === 'papers' && filters.area && filters.area.length > 0) {
        currentFilters.area = filters.area; // 传递数组
    }

    // --- 添加新的过滤器到 currentFilters (修正参数名和值) --- 
    if (searchTarget === 'models') {
      if (filters.library_name) currentFilters.filter_library_name = filters.library_name; // 修正参数名
      if (filters.tags && filters.tags.length > 0) currentFilters.filter_tags = filters.tags; // 修正参数名，传递数组 (假设 useModelSearch 能处理)
      if (filters.author) currentFilters.filter_author = filters.author; // 修正参数名
    }
    if (searchTarget === 'papers') {
      // 修正：使用 filter_authors 参数名，并假设 filters.paper_author 是 string[]
      if (filters.paper_author && filters.paper_author.length > 0) currentFilters.filter_authors = filters.paper_author;
    }
    // --- 结束添加 ---

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

  // --- 处理错误信息 ---
  const errorMessage = useMemo(() => {
    if (!isError || !error) return null;
    
    // 尝试从 React Query 的 error 对象中提取后端返回的 detail
    // 注意：错误结构可能因 axios 或 fetch 的包装而异
    let detail = '搜索时发生未知错误，请稍后再试。'; // 默认错误信息
    if (error instanceof Error) {
      // 尝试解析常见的 Axios 或 Fetch 错误结构
      // @ts-ignore - 尝试访问可能存在的 response 属性
      const responseData = error.response?.data;
      if (responseData && typeof responseData === 'object' && 'detail' in responseData) {
        // @ts-ignore
        detail = responseData.detail || detail;
      // @ts-ignore
      } else if (error.response?.statusText) {
        // @ts-ignore
        detail = `错误：${error.response.status} - ${error.response.statusText}`;
      } else {
        detail = error.message; // 如果无法解析，使用原始错误消息
      }
    }
    return detail;
  }, [isError, error]);

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
    const newPage = 1;
    setRawQuery(newQuery);
    setCurrentPage(newPage);
    updateSearchParams({ query: newQuery, page: newPage });
  };

  // --- 处理目标改变 ---
  const handleTargetChange = (newTarget: ActualSearchTarget) => {
    // 在函数内部进行检查和类型断言，确保我们处理的是 'papers' 或 'models'
    if (newTarget !== 'papers' && newTarget !== 'models') {
        console.warn(`Received unexpected target: ${newTarget}, ignoring.`);
        return;
    }
    // 类型断言仍然安全，因为我们已经检查过
    const validTarget = newTarget as SearchTarget; 

    const newPage = 1;
    const newSort = 'score';
    const newFilters = {};
    setSearchTarget(validTarget); // 使用断言后的类型
    setCurrentPage(newPage);
    setSortBy(newSort);
    setFilters(newFilters);
    updateSearchParams({ target: validTarget, page: newPage, sort: newSort, filters: newFilters, query: rawQuery, mode: searchMode });
  };

  // --- 处理排序改变 ---
  const handleSortChange = (newSortOption: string) => {
    const newPage = 1;
    setSortBy(newSortOption);
    setCurrentPage(newPage);
    updateSearchParams({ sort: newSortOption, page: newPage });
  };

  // --- 处理过滤改变 ---
  const handleFilterChange = (newFilters: Filters) => {
    const newPage = 1;
    setFilters(newFilters);
    setCurrentPage(newPage);
    updateSearchParams({ filters: newFilters, page: newPage });
  };

  // --- 处理分页改变 ---
  const handlePageChange = (page: number) => {
    setCurrentPage(page);
    updateSearchParams({ page: page });
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

  // --- Helper to Update URL Search Params --- 
  const updateSearchParams = (newParams: {
    query?: string;
    mode?: SearchMode;
    target?: SearchTarget;
    sort?: string;
    page?: number;
    filters?: Filters;
  }) => {
    const params = new URLSearchParams(searchParams);

    // Update query (q)
    if (newParams.query !== undefined) {
      if (newParams.query) params.set('q', newParams.query);
      else params.delete('q');
    }
    // Update mode
    if (newParams.mode !== undefined) {
      if (newParams.mode !== 'keyword') params.set('mode', newParams.mode);
      else params.delete('mode');
    }
    // Update target
    if (newParams.target !== undefined) {
        if (newParams.target !== 'papers') params.set('target', newParams.target);
        else params.delete('target');
    }
    // Update sort
    if (newParams.sort !== undefined) {
      if (newParams.sort !== 'score') params.set('sort', newParams.sort);
      else params.delete('sort');
    }
    // Update page
    if (newParams.page !== undefined) {
      if (newParams.page > 1) params.set('page', newParams.page.toString());
      else params.delete('page');
    }
    // Update filters
    if (newParams.filters !== undefined) {
      // First, remove all existing filter params
      // (Simpler than tracking individual filter changes)
      params.delete('date_from');
      params.delete('date_to');
      params.delete('area');
      params.delete('pipeline_tag');
      params.delete('filter_library_name');
      params.delete('filter_tags');
      params.delete('filter_author');
      params.delete('filter_authors');
      
      // Add new filter params if they exist
      const currentFilters = newParams.filters;
      if (currentFilters.year_from) params.set('date_from', `${currentFilters.year_from}-01-01`);
      if (currentFilters.year_to) params.set('date_to', `${currentFilters.year_to}-12-31`);
      if (currentFilters.area && currentFilters.area.length > 0) params.set('area', currentFilters.area.join(','));
      if (currentFilters.pipeline_tag) params.set('pipeline_tag', currentFilters.pipeline_tag);
      if (currentFilters.library_name) params.set('filter_library_name', currentFilters.library_name);
      if (currentFilters.tags && currentFilters.tags.length > 0) params.set('filter_tags', currentFilters.tags.join(','));
      if (currentFilters.author) params.set('filter_author', currentFilters.author);
      if (currentFilters.paper_author && currentFilters.paper_author.length > 0) params.set('filter_authors', currentFilters.paper_author.join(','));
    }

    setSearchParams(params, { replace: true }); // Use replace to avoid polluting history too much
  };

  // --- Modified State Update Handlers ---

  const handleModeChange = (newMode: SearchMode) => {
    const newPage = 1;
    setSearchMode(newMode);
    setCurrentPage(newPage);
    updateSearchParams({ mode: newMode, page: newPage });
  };

  return (
    <div className="container mx-auto p-4 md:p-6 lg:p-8 max-w-7xl">
      <header className="mb-6">
        <h1 className="text-3xl md:text-4xl font-bold text-gray-800 text-center mb-2">
          AIGraphX 探索平台
        </h1>
        <p className="text-md text-gray-600 text-center">
          发现、关联和理解 AI 领域的最新进展
        </p>
      </header>

      {/* 搜索栏 */}
      <div className="mb-6">
        <SearchBar initialQuery={rawQuery} onSearch={handleSearch} />
      </div>

      {/* 全局图谱跳转入口 - 新增区域 */}
      <div className="my-6 p-4 bg-indigo-50 rounded-lg shadow hover:shadow-md transition-shadow">
        <Link to="/global-graph" className="flex items-center text-indigo-700 hover:text-indigo-900 group">
          <GlobeAltIcon className="h-8 w-8 mr-3 text-indigo-500 group-hover:text-indigo-700 transition-colors" />
          <div>
            <h3 className="text-lg font-semibold">探索全局知识图谱</h3>
            <p className="text-sm text-indigo-600 group-hover:text-indigo-800">
              可视化所有实体及其关联，发现隐藏的连接。
            </p>
          </div>
        </Link>
      </div>

      {/* 搜索控制栏: 模式，目标，排序 */}
      <div className="flex flex-col md:flex-row justify-between items-center mb-6 gap-4 p-4 bg-gray-50 rounded-lg shadow-sm">
        <div className="flex flex-col sm:flex-row gap-4 w-full md:w-auto">
          <SearchModeToggle searchMode={searchMode} onChange={handleModeChange} />
          <TargetSelector searchTarget={searchTarget} onChange={handleTargetChange} />
        </div>
        <div className="w-full md:w-auto md:min-w-[200px]">
          <SortDropdown
            options={currentSortOptions}
            selectedOption={sortBy}
            onChange={handleSortChange}
          />
        </div>
      </div>
      
      {/* 主内容区: 过滤器 + 结果列表 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6"> 
        {/* Left Column: Filters */} 
        <div className="md:col-span-1">
          <FilterPanel
            initialFilters={filters}
            onFilterChange={handleFilterChange}
            target={searchTarget}
          />
        </div>

        {/* Right Column: Results and Pagination */} 
        <div className="md:col-span-3 space-y-4"> 
          {isError && (
            <div className="text-red-600 bg-red-100 p-4 rounded-md">
              搜索出错: {errorMessage}
            </div>
          )}

          {isLoading ? (
            <div className="text-center py-10">加载中...</div>
          ) : searchResult && searchResult.items && searchResult.items.length > 0 ? (
            <>
              <ResultList
                results={results}
                target={searchTarget}
                isLoading={isLoading || isFetching}
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
          )}
        </div>
      </div>
    </div>
  );
};

export default SearchPage; 