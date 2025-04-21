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

// TODO: 添加 useModelSearch, usePaperDetail, useModelDetail 等 Hook 