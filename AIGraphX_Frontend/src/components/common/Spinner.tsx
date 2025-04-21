import React from 'react';

interface SpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  color?: string; // e.g., 'border-blue-600'
  className?: string;
}

const Spinner: React.FC<SpinnerProps> = ({
  size = 'md',
  color = 'border-blue-600',
  className = '',
}) => {
  const sizeStyles = {
    sm: 'h-6 w-6',
    md: 'h-10 w-10',
    lg: 'h-16 w-16',
  };

  const spinnerStyles = `animate-spin rounded-full border-b-2 ${sizeStyles[size]} ${color} ${className}`;

  return <div className={spinnerStyles}></div>;
};

export default Spinner; 