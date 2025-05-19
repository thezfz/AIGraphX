import { useQuery } from '@tanstack/react-query';
import apiClient from './apiClient';
import type { paths, components } from '../types/api'; // 导入生成的类型

// 类型别名，方便使用
type PaperSearchQuery = paths['/api/v1/search/papers/']['get']['parameters']['query'];
type PaginatedPaperResult = components['schemas']['PaginatedPaperSearchResult'];
// 注意: OpenAPI 生成器有时会对联合类型响应体处理不佳，这里我们显式指定 Paper 类型
// 如果您的 API 总是返回 PaginatedPaperSearchResult，可以直接使用它。
// 如果可能返回 PaginatedSemanticSearchResult，则需要进一步处理或调整类型。

/**
 * 获取论文搜索结果的 React Query Hook。
 * 
 * @param params - 搜索参数对象 (q, search_type, skip, limit, sort_by, sort_order, etc.)
 * @returns React Query 查询结果对象
 */
export const usePaperSearch = (params: PaperSearchQuery) => {
  return useQuery<
    PaginatedPaperResult, // 成功时的数据类型
    Error // 错误类型
  >(
    {
      // 查询键：包含所有会影响查询结果的参数
      queryKey: ['searchPapers', params],
      // 查询函数：使用 apiClient 发起 GET 请求
      queryFn: async () => {
        const response = await apiClient.get<PaginatedPaperResult>('/search/papers/', {
          params: {
            ...params,
            skip: params.skip ?? 0, // 提供默认值
            limit: params.limit ?? 10, // 提供默认值
          },
        });
        return response.data;
      },
      // 选项
      enabled: !!params.q, // 仅当查询字符串 (q) 存在时才执行查询
      placeholderData: (previousData) => previousData, // 在新数据加载时保留旧数据 (v5 替代 keepPreviousData)
      // staleTime: 5 * 60 * 1000, // 数据在5分钟内保持新鲜 (可选)
      // cacheTime: 10 * 60 * 1000, // 数据在10分钟内保留在缓存中 (可选)
    }
  );
};

/**
 * 获取可用论文领域列表的 React Query Hook。
 * 
 * @returns React Query 查询结果对象，包含所有可用的论文领域
 */
export const usePaperAreas = () => {
  return useQuery<string[], Error>({
    queryKey: ['paperAreas'],
    queryFn: async () => {
      const response = await apiClient.get<string[]>('/search/paper-areas/');
      return response.data;
    },
    staleTime: 24 * 60 * 60 * 1000, // 数据保持新鲜24小时，因为领域列表变化不频繁
  });
};

// TODO: 添加 useModelSearch, usePaperDetail, useModelDetail 等 Hook

// --- Graph Data Hooks ---

/** GraphData 类型别名 */
type GraphData = components['schemas']['GraphData'];

/**
 * 获取论文图数据的 React Query Hook。
 * 
 * @param pwcId - 论文的 PWC ID。
 * @param enabled - 是否启用查询，默认为 true。如果 pwcId 为空则通常不启用。
 * @returns React Query 查询结果对象，包含图数据。
 */
export const usePaperGraphData = (pwcId: string, enabled: boolean = true) => {
  return useQuery<
    GraphData, 
    Error
  >(
    {
      queryKey: ['paperGraph', pwcId],
      queryFn: async () => {
        const response = await apiClient.get<GraphData>(`/graph/papers/${pwcId}/graph`);
        return response.data;
      },
      enabled: !!pwcId && enabled, // 仅当 pwcId 存在且外部允许时才执行查询
    }
  );
};

/**
 * 获取模型图数据的 React Query Hook。
 * 
 * @param modelId - 模型的 ID。
 * @param enabled - 是否启用查询，默认为 true。如果 modelId 为空则通常不启用。
 * @returns React Query 查询结果对象，包含图数据。
 */
export const useModelGraphData = (
  modelId: string,
  enabled: boolean = true // Allow enabling/disabling the query
) => {
  return useQuery<
    GraphData,
    Error // Specify Error type for TanStack Query v5
  >({ 
    queryKey: ['modelGraph', modelId],
    queryFn: () => fetchModelGraphData(modelId),
    enabled: enabled && !!modelId, // Only enable if modelId is truthy and enabled prop is true
    staleTime: 1000 * 60 * 5, // 5 minutes
  });
};

// New fetch function for radial focus graph data
export const fetchModelGraphData = async (modelId: string): Promise<GraphData> => {
  const response = await apiClient.get(`/graph/hf_models/${modelId}/graph`);
  return response.data;
};

// New fetch function for radial focus graph data
export const fetchModelRadialFocusGraphData = async (modelId: string): Promise<GraphData> => {
  const response = await apiClient.get(`/graph/hf_models/${modelId}/radial_focus`); // Correct endpoint
  return response.data;
};

// New hook for fetching radial focus graph data
export const useModelRadialFocusGraphData = (
  modelId: string,
  enabled: boolean = true
) => {
  return useQuery<
    GraphData, 
    Error
  >({
    queryKey: ['modelRadialFocusGraph', modelId],
    queryFn: () => fetchModelRadialFocusGraphData(modelId), // Correctly calls its own fetcher
    enabled: enabled && !!modelId,
    staleTime: 1000 * 60 * 5, // 5 minutes
  });
};

// --- Detail Hooks ---
/** ModelDetail 类型别名 */
type ModelDetail = components['schemas']['HFModelDetail']; // 确保 HFModelDetail 在 api.ts 中定义正确

/**
 * 获取模型详细信息的 React Query Hook。
 * 
 * @param modelId - 模型的 ID。
 * @param enabled - 是否启用查询，默认为 true。如果 modelId 为空则通常不启用。
 * @returns React Query 查询结果对象，包含模型详细信息。
 */
export const useModelDetail = (modelId?: string, enabled: boolean = true) => { // modelId 可以是 undefined
  return useQuery<
    ModelDetail, 
    Error
  >(
    {
      queryKey: ['modelDetail', modelId],
      queryFn: async () => {
        if (!modelId) { // 如果 modelId 未定义，则不执行查询并抛出错误或返回特定值
          throw new Error('Model ID is undefined, cannot fetch details.');
          // 或者 return Promise.resolve(null); 并相应调整类型
        }
        const response = await apiClient.get<ModelDetail>(`/graph/models/${modelId}`);
        return response.data;
      },
      enabled: !!modelId && enabled, // 仅当 modelId 存在且外部允许时才执行查询
    }
  );
};

/** PaperDetail 类型别名 */
type PaperDetail = components['schemas']['PaperDetailResponse']; // 确保 PaperDetailResponse 在 api.ts 中定义正确

/**
 * 获取论文详细信息的 React Query Hook。
 *
 * @param pwcId - 论文的 PWC ID。
 * @param enabled - 是否启用查询，默认为 true。如果 pwcId 为空则通常不启用。
 * @returns React Query 查询结果对象，包含论文详细信息。
 */
export const usePaperDetail = (pwcId?: string, enabled: boolean = true) => { // pwcId 可以是 undefined
  return useQuery<
    PaperDetail,
    Error
  >(
    {
      queryKey: ['paperDetail', pwcId],
      queryFn: async () => {
        if (!pwcId) { // 如果 pwcId 未定义，则不执行查询
          throw new Error('PWC ID is undefined, cannot fetch paper details.');
        }
        const response = await apiClient.get<PaperDetail>(`/graph/papers/${pwcId}`);
        return response.data;
      },
      enabled: !!pwcId && enabled, // 仅当 pwcId 存在且外部允许时才执行查询
    }
  );
};

// New hook to fetch all graph data
export const useAllGraphData = () => {
  const fetchAllGraphData = async (): Promise<GraphData> => {
    const response = await apiClient.get<GraphData>('/graph/all-data');
    return response.data;
  };

  return useQuery<GraphData, Error>({
    queryKey: ['allGraphData'], 
    queryFn: fetchAllGraphData,
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes (renamed from cacheTime in v5)
    // Add other options like onError, onSuccess, enabled, etc. as needed
  });
}; 