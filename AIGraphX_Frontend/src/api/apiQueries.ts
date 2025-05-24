import { useQuery } from '@tanstack/react-query';
import apiClient from './apiClient';
import type { paths, components } from '../types/api'; // 确保路径正确

// --- Search Parameters Interfaces (from src/api/apiQueries.ts) --- //
interface BaseSearchParams {
  query: string;
  skip?: number;
  limit?: number;
  search_type?: 'semantic' | 'keyword' | 'hybrid';
}

export interface PaperSearchParams extends BaseSearchParams {
  sort_by?: paths['/api/v1/search/papers/']['get']['parameters']['query']['sort_by'];
  sort_order?: 'asc' | 'desc';
  date_from?: string | null;
  date_to?: string | null;
  area?: string[] | null; // Changed to string[] for multi-select, and aligned with potential backend
  // Other paper-specific filters can be added here
}

export interface ModelSearchParams extends BaseSearchParams {
  sort_by?: paths['/api/v1/search/models/']['get']['parameters']['query']['sort_by'];
  sort_order?: 'asc' | 'desc';
  pipeline_tag?: string | null;
  library_name?: string | null;
  tags?: string[] | null; // Changed to string[]
  author?: string | null;
  // Other model-specific filters can be added here
}


// --- API Response and Schema Types (Consolidated) --- //
type PaginatedPaperSearchResult = components['schemas']['PaginatedPaperSearchResult'];
// TODO: Verify and use the correct type for PaginatedModelSearchResult from OpenAPI schema if available.
// For now, using a generic or a placeholder if it causes issues, or assuming it's similar to Paper's for structure.
type PaginatedModelSearchResult = components['schemas']['PaginatedPaperSearchResult']; // Placeholder, ensure this type exists or use a more generic one.
type PaperDetailResponse = components['schemas']['PaperDetailResponse'];
type ModelDetailResponse = components['schemas']['HFModelDetail']; // Corrected from services
type GraphData = components['schemas']['GraphData'];

const STALE_TIME = 5 * 60 * 1000; // 5 minutes
const LONG_STALE_TIME = 24 * 60 * 60 * 1000; // 24 hours

// --- Helper Function for Cleaning Params (from src/api/apiQueries.ts) --- //
const cleanParams = (params: Record<string, any>): Record<string, any> => {
  const cleaned: Record<string, any> = {};
  for (const key in params) {
    if (Object.prototype.hasOwnProperty.call(params, key)) {
      let value = params[key];
      // Convert empty arrays to null to be removed, or handle as per backend expectation
      if (Array.isArray(value) && value.length === 0) {
        value = null;
      }
      if (value !== null && value !== undefined && value !== '') {
        if (key === 'query') {
          cleaned['q'] = value;
        } else {
          cleaned[key] = value;
        }
      }
    }
  }
  return cleaned;
};

// --- React Query Hooks (Consolidated and Enhanced) --- //

/**
 * Hook for searching papers.
 * (Enhanced with default skip/limit and placeholderData from src/services)
 */
export const usePaperSearch = (params: PaperSearchParams, enabled: boolean = true) => {
  return useQuery<PaginatedPaperSearchResult, Error>({
    queryKey: ['paperSearch', params],
    queryFn: async () => {
      const apiParams = cleanParams({
        ...params,
        skip: params.skip ?? 0,
        limit: params.limit ?? 10,
      });
      // Ensure the endpoint matches what's used in services or is correct
      const response = await apiClient.get<PaginatedPaperSearchResult>('/search/papers/', { params: apiParams });
      return response.data;
    },
    enabled: enabled && !!params.query,
    staleTime: STALE_TIME,
    placeholderData: (previousData: any) => previousData,
  });
};

/**
 * Hook for searching models.
 * (New or significantly different from src/api, similar to a placeholder in src/services TODO)
 * This assumes a similar structure to usePaperSearch.
 */
export const useModelSearch = (params: ModelSearchParams, enabled: boolean = true) => {
  return useQuery<PaginatedModelSearchResult, Error>({
    queryKey: ['modelSearch', params],
    queryFn: async () => {
      const apiParams = cleanParams({
        ...params,
        skip: params.skip ?? 0,
        limit: params.limit ?? 10,
      });
      const response = await apiClient.get<PaginatedModelSearchResult>('/search/models/', { params: apiParams });
      return response.data;
    },
    enabled: enabled && !!params.query,
    staleTime: STALE_TIME,
    placeholderData: (previousData: any) => previousData,
  });
};


/**
 * Hook for getting paper details. (Based on src/api, path corrected in previous steps)
 */
export const usePaperDetail = (pwcId?: string, enabled: boolean = true) => {
  return useQuery<PaperDetailResponse, Error>({
    queryKey: ['paperDetail', pwcId],
    queryFn: async () => {
      if (!pwcId) {
        throw new Error('PWC ID is required for paper detail');
      }
      const response = await apiClient.get<PaperDetailResponse>(`/graph/papers/${pwcId}`);
      return response.data;
    },
    enabled: enabled && !!pwcId,
    staleTime: STALE_TIME,
  });
};

/**
 * Hook for getting model details. (Based on src/api, with console.logs removed later)
 * Path already corrected, retry logic kept.
 */
export const useModelDetail = (modelId?: string, enabled: boolean = true) => {
  return useQuery<ModelDetailResponse, Error>({
    queryKey: ['modelDetail', modelId],
    queryFn: async () => {
      if (!modelId) {
        throw new Error('Model ID is required');
      }
      const encodedModelId = encodeURIComponent(modelId);
      const response = await apiClient.get<ModelDetailResponse>(`/graph/models/${encodedModelId}`);
      return response.data;
    },
    enabled: enabled && !!modelId,
    staleTime: STALE_TIME,
    retry: 3,
    retryDelay: attempt => Math.min(1000 * 2 ** attempt, 30000),
  });
};

// --- Hooks from src/services/apiQueries.ts ---

/**
 * 获取可用论文领域列表的 React Query Hook。
 */
export const usePaperAreas = () => {
  return useQuery<string[], Error>({
    queryKey: ['paperAreas'],
    queryFn: async () => {
      // Corrected endpoint based on typical structure, assuming /api/v1 prefix from apiClient
      const response = await apiClient.get<string[]>('/search/paper-areas/');
      return response.data;
    },
    staleTime: LONG_STALE_TIME, // Use the defined long stale time
  });
};

/**
 * 获取论文图数据的 React Query Hook。
 */
export const usePaperGraphData = (pwcId?: string, enabled: boolean = true) => {
  return useQuery<GraphData, Error>({
    queryKey: ['paperGraph', pwcId],
    queryFn: async () => {
      if (!pwcId) throw new Error("PWC ID is required to fetch paper graph data.");
      const response = await apiClient.get<GraphData>(`/graph/papers/${pwcId}/graph`);
      return response.data;
    },
    enabled: !!pwcId && enabled,
    staleTime: STALE_TIME,
  });
};

/**
 * Helper fetch function for model graph data (kept from src/services)
 */
export const fetchModelGraphData = async (modelId: string): Promise<GraphData> => {
  // Assuming modelId might contain special characters, though not strictly necessary if useModelGraphData handles it
  const encodedModelId = encodeURIComponent(modelId); 
  const response = await apiClient.get<GraphData>(`/graph/hf_models/${encodedModelId}/graph`);
  return response.data;
};

/**
 * 获取模型图数据的 React Query Hook。
 */
export const useModelGraphData = (modelId?: string, enabled: boolean = true) => {
  return useQuery<GraphData, Error>({
    queryKey: ['modelGraph', modelId],
    queryFn: async () => {
      if (!modelId) throw new Error("Model ID is required to fetch model graph data.");
      return fetchModelGraphData(modelId); // Calls the separate fetcher
    },
    enabled: !!modelId && enabled,
    staleTime: STALE_TIME,
  });
};

/**
 * New hook to fetch all graph data (from src/services)
 */
export const useAllGraphData = (enabled: boolean = true) => { // Added enabled flag
  return useQuery<GraphData, Error>({
    queryKey: ['allGraphData'],
    queryFn: async () => {
        const response = await apiClient.get<GraphData>('/graph/all-data');
        return response.data;
    },
    enabled: enabled, // Controlled by the enabled flag
    staleTime: STALE_TIME, 
    gcTime: 10 * 60 * 1000, 
  });
};

// Note: useModelRadialFocusGraphData and its fetcher are intentionally omitted as per instructions. 