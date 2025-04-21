import React, { useState, useRef, useEffect } from 'react';

export type SortOption = {
  value: string;
  label: string;
};

interface SortDropdownProps {
  options: SortOption[];
  selectedOption: string;
  onChange: (value: string) => void;
  label?: string;
}

const SortDropdown: React.FC<SortDropdownProps> = ({
  options,
  selectedOption,
  onChange,
  label = '排序方式',
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // 获取当前选中的选项标签
  const selectedLabel = options.find((option) => option.value === selectedOption)?.label || '';

  // 切换下拉列表打开/关闭状态
  const toggleDropdown = () => setIsOpen(!isOpen);

  // 处理选项选择
  const handleSelect = (value: string) => {
    onChange(value);
    setIsOpen(false);
  };

  // 点击外部关闭下拉列表
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  return (
    <div className="relative" ref={dropdownRef}>
      <span className="mr-2 text-gray-700">{label}:</span>
      <button
        type="button"
        className="inline-flex justify-between items-center w-40 px-3 py-1 bg-white border border-gray-300 rounded-md text-sm text-gray-700 hover:bg-gray-50"
        onClick={toggleDropdown}
      >
        <span>{selectedLabel}</span>
        <svg
          className={`ml-1 h-5 w-5 transition-transform ${isOpen ? 'transform rotate-180' : ''}`}
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 20 20"
          fill="currentColor"
        >
          <path
            fillRule="evenodd"
            d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
            clipRule="evenodd"
          />
        </svg>
      </button>

      {isOpen && (
        <div className="absolute mt-1 w-full bg-white border border-gray-200 rounded-md shadow-lg z-10">
          <ul className="py-1">
            {options.map((option) => (
              <li key={option.value}>
                <button
                  type="button"
                  className={`block w-full text-left px-4 py-2 text-sm ${
                    option.value === selectedOption
                      ? 'bg-blue-100 text-blue-800'
                      : 'text-gray-700 hover:bg-gray-100'
                  }`}
                  onClick={() => handleSelect(option.value)}
                >
                  {option.label}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default SortDropdown; 