import { useQuery } from '@tanstack/react-query';
import apiClient from './apiClient';
import type { paths } from '../types/api'; // 确保路径正确

// --- Search Parameters (Extending with filters) --- //
interface BaseSearchParams {
  query: string;
  skip?: number;
  limit?: number;
  search_type?: 'semantic' | 'keyword' | 'hybrid'; // Added search_type
}

export interface PaperSearchParams extends BaseSearchParams {
  sort_by?: paths['/api/v1/search/papers/']['get']['parameters']['query']['sort_by'];
  sort_order?: 'asc' | 'desc';
  date_from?: string | null; // Added date filters (YYYY-MM-DD)
  date_to?: string | null;
  area?: string | null; // Example: Add area filter
}

export interface ModelSearchParams extends BaseSearchParams {
  sort_by?: paths['/api/v1/search/models/']['get']['parameters']['query']['sort_by'];
  sort_order?: 'asc' | 'desc';
  // Add model specific filters here, e.g.:
  // task?: string | null;
}

// --- API Response Types (ensure these match your openapi-typescript output) --- //
type PaperSearchResponse = paths['/api/v1/search/papers/']['get']['responses']['200']['content']['application/json'];
type ModelSearchResponse = paths['/api/v1/search/models/']['get']['responses']['200']['content']['application/json'];
// Corrected paths based on types/api.ts
type PaperDetailResponse = paths['/api/v1/graph/papers/{pwc_id}']['get']['responses']['200']['content']['application/json'];
type ModelDetailResponse = paths['/api/v1/graph/models/{model_id}']['get']['responses']['200']['content']['application/json'];

const STALE_TIME = 5 * 60 * 1000; // 5 minutes

// --- Helper Function for Cleaning Params --- //

const cleanParams = (params: Record<string, any>): Record<string, any> => {
  const cleaned: Record<string, any> = {};
  for (const key in params) {
    if (Object.prototype.hasOwnProperty.call(params, key)) {
      const value = params[key];
      if (value !== null && value !== undefined && value !== '') { // Also remove empty strings
        if (key === 'query') {
          cleaned['q'] = value; // Rename query to q
        } else {
          cleaned[key] = value;
        }
      }
    }
  }
  return cleaned;
};

// --- React Query Hooks (Restoring cleanParams) --- //

// Hook for searching papers
export const usePaperSearch = (params: PaperSearchParams, enabled: boolean = true) => {
  return useQuery<PaperSearchResponse, Error>({
    queryKey: ['paperSearch', params], 
    queryFn: async () => {
      // Use the cleanParams helper function
      const apiParams = cleanParams(params);

      const response = await apiClient.get<PaperSearchResponse>('/api/v1/search/papers/', {
        params: apiParams, // Send cleaned parameters
      });
      return response.data;
    },
    enabled: enabled && !!params.query, // Enable only if query is present
    staleTime: STALE_TIME,
  });
};

// Hook for searching models
export const useModelSearch = (params: ModelSearchParams, enabled: boolean = true) => {
  return useQuery<ModelSearchResponse, Error>({
    queryKey: ['modelSearch', params],
    queryFn: async () => {
      // Use the cleanParams helper function
      const apiParams = cleanParams(params);

      const response = await apiClient.get<ModelSearchResponse>('/api/v1/search/models/', {
        params: apiParams, // Send cleaned parameters
      });
      return response.data;
    },
    enabled: enabled && !!params.query, // Enable only if query is present
    staleTime: STALE_TIME,
  });
};

// Hook for getting paper details
// Updated to use pwcId and correct path
export const usePaperDetail = (pwcId: string | undefined, enabled: boolean = true) => {
  return useQuery<PaperDetailResponse, Error>({
    // Use a more specific query key including the ID
    queryKey: ['paperDetail', pwcId], // Use pwcId in queryKey
    queryFn: async () => {
      if (!pwcId) {
        throw new Error('PWC ID is required for paper detail'); // Updated error message
      }
      // Use the correct API path from types/api.ts
      const response = await apiClient.get<PaperDetailResponse>(`/api/v1/graph/papers/${pwcId}`);
      return response.data;
    },
    // Only run if enabled and pwcId exists
    enabled: enabled && !!pwcId,
    staleTime: STALE_TIME,
  });
};

// Hook for getting model details
// Updated path to match types/api.ts
export const useModelDetail = (modelId: string | undefined, enabled: boolean = true) => {
  return useQuery<ModelDetailResponse, Error>({
    queryKey: ['modelDetail', modelId],
    queryFn: async () => {
      if (!modelId) {
        throw new Error('Model ID is required');
      }
      
      // 使用 encodeURIComponent 确保特殊字符（如/）被正确编码
      const encodedModelId = encodeURIComponent(modelId);
      
      // 日志输出，帮助调试
      console.log(`Fetching model details for ID: ${modelId}`);
      console.log(`Encoded URL: /api/v1/graph/models/${encodedModelId}`);
      
      try {
        // Use the correct API path with encoded ID
        const response = await apiClient.get<ModelDetailResponse>(`/api/v1/graph/models/${encodedModelId}`);
        console.log('Model detail response:', response.data);
        return response.data;
      } catch (error) {
        console.error('Error fetching model details:', error);
        throw error;
      }
    },
    enabled: enabled && !!modelId,
    staleTime: STALE_TIME,
    // 添加重试策略
    retry: 3,
    retryDelay: attempt => Math.min(1000 * 2 ** attempt, 30000),
  });
}; 