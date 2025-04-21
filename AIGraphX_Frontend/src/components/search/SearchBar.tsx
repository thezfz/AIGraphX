import React, { useState } from 'react';

interface SearchBarProps {
  onSearch: (query: string) => void;
  placeholder?: string;
  initialQuery?: string;
  isLoading?: boolean;
}

const SearchBar: React.FC<SearchBarProps> = ({
  onSearch,
  placeholder = '输入关键字或描述...',
  initialQuery = '',
  isLoading = false,
}) => {
  const [query, setQuery] = useState<string>(initialQuery);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="flex flex-col md:flex-row gap-4">
        <div className="flex-grow">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={placeholder}
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        
        <button
          type="submit"
          className="bg-blue-600 text-white px-6 py-2 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-blue-400 disabled:cursor-not-allowed"
          disabled={isLoading}
        >
          {isLoading ? '搜索中...' : '搜索'}
        </button>
      </div>
    </form>
  );
};

export default SearchBar; 