import { useState, useEffect } from 'react';

/**
 * Debounces a value.
 * @param value The value to debounce.
 * @param delay The debounce delay in milliseconds.
 * @returns The debounced value.
 */
export function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    // 设置一个定时器，在 delay 毫秒后更新 debounced value
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    // 清理函数：在 value 或 delay 变化时，或者组件卸载时，清除之前的定时器
    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]); // 仅当 value 或 delay 变化时重新设置定时器

  return debouncedValue;
} 