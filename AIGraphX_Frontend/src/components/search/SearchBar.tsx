import React, { useState, useEffect, useRef } from 'react';

interface SearchBarProps {
  onSearch: (query: string) => void;
  placeholder?: string;
  initialQuery?: string;
  isLoading?: boolean;
}

const LOCAL_STORAGE_KEY = 'searchHistory';
const MAX_HISTORY_LENGTH = 10;

const SearchBar: React.FC<SearchBarProps> = ({
  onSearch,
  placeholder = '输入关键字或描述...',
  initialQuery = '',
  isLoading = false,
}) => {
  const [query, setQuery] = useState<string>(initialQuery);
  const [history, setHistory] = useState<string[]>([]);
  const [showHistory, setShowHistory] = useState<boolean>(false);
  const searchBarRef = useRef<HTMLDivElement>(null); // Ref for detecting clicks outside

  // Load history from localStorage on mount
  useEffect(() => {
    const storedHistory = localStorage.getItem(LOCAL_STORAGE_KEY);
    if (storedHistory) {
      try {
        setHistory(JSON.parse(storedHistory));
      } catch (e) {
        console.error("Failed to parse search history from localStorage", e);
        localStorage.removeItem(LOCAL_STORAGE_KEY); // Clear corrupted data
      }
    }
  }, []);

  // Effect to handle clicks outside the search bar to close history
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (searchBarRef.current && !searchBarRef.current.contains(event.target as Node)) {
        setShowHistory(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const saveToHistory = (searchTerm: string) => {
    const term = searchTerm.trim();
    if (!term) return;

    setHistory(prevHistory => {
        // Remove duplicates and add the new term to the beginning
        const updatedHistory = [term, ...prevHistory.filter(item => item !== term)];
        // Limit the history length
        const limitedHistory = updatedHistory.slice(0, MAX_HISTORY_LENGTH);

        // Save to localStorage
        localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(limitedHistory));
        return limitedHistory;
    });
  };

  const handleSearch = (searchTerm: string) => {
    const term = searchTerm.trim();
    if (term) {
      saveToHistory(term); // Save before performing search
      onSearch(term);
      setShowHistory(false); // Hide history after search
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleSearch(query);
  };

  const handleHistoryClick = (term: string) => {
    setQuery(term);
    handleSearch(term);
  };

  const handleInputFocus = () => {
    if (history.length > 0) {
        setShowHistory(true);
    }
  };

  // Use onMouseDown instead of onClick for history items
  // to prevent onBlur from firing before the click is registered.
  const handleHistoryMouseDown = (term: string) => {
    handleHistoryClick(term);
  };

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="flex flex-col md:flex-row gap-4" ref={searchBarRef}>
        <div className="flex-grow relative">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={handleInputFocus}
            placeholder={placeholder}
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            aria-autocomplete="list"
            aria-controls="search-history-list"
          />
          {showHistory && history.length > 0 && (
            <div 
              id="search-history-list"
              className="absolute top-full left-0 right-0 mt-1 bg-white border border-gray-300 rounded-md shadow-lg z-10 max-h-60 overflow-y-auto"
            >
              <ul role="listbox" aria-label="搜索历史">
                {history.map((term, index) => (
                  <li key={index} role="option" aria-selected="false">
                    <button
                      type="button"
                      onMouseDown={() => handleHistoryMouseDown(term)} 
                      className="w-full text-left px-4 py-2 hover:bg-gray-100 focus:outline-none focus:bg-gray-100 cursor-pointer"
                    >
                      {term}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
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