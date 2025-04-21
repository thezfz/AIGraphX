import React, { InputHTMLAttributes } from 'react';

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  containerClassName?: string;
}

const Input: React.FC<InputProps> = ({
  label,
  error,
  id,
  className = '',
  containerClassName = '',
  ...props
}) => {
  const inputId = id || (label ? label.toLowerCase().replace(/\s+/g, '-') : undefined);

  const baseStyles = "w-full px-4 py-2 border rounded-md focus:outline-none focus:ring-2";
  const borderStyles = error ? "border-red-500 focus:ring-red-500" : "border-gray-300 focus:ring-blue-500";
  const inputStyles = `${baseStyles} ${borderStyles} ${className}`;

  return (
    <div className={`w-full ${containerClassName}`}>
      {label && (
        <label htmlFor={inputId} className="block text-sm font-medium text-gray-700 mb-1">
          {label}
        </label>
      )}
      <input
        id={inputId}
        className={inputStyles}
        {...props}
      />
      {error && <p className="mt-1 text-xs text-red-600">{error}</p>}
    </div>
  );
};

export default Input; 