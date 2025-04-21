import React from 'react';

interface PaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  maxVisiblePages?: number; // 控制显示多少个页码按钮（包括省略号）
}

const Pagination: React.FC<PaginationProps> = ({
  currentPage,
  totalPages,
  onPageChange,
  maxVisiblePages = 7, // 默认显示7个页码按钮
}) => {
  if (totalPages <= 1) {
    return null; // 如果只有一页或没有页，则不显示分页
  }

  const handlePageClick = (page: number) => {
    if (page >= 1 && page <= totalPages && page !== currentPage) {
      onPageChange(page);
    }
  };

  const renderPageNumbers = () => {
    const pageNumbers: (number | string)[] = [];
    const halfVisible = Math.floor(maxVisiblePages / 2);

    if (totalPages <= maxVisiblePages) {
      // 总页数小于等于最大可见页数，显示所有页码
      for (let i = 1; i <= totalPages; i++) {
        pageNumbers.push(i);
      }
    } else {
      // 总页数大于最大可见页数，需要省略号
      pageNumbers.push(1);

      let startPage = Math.max(2, currentPage - halfVisible + (maxVisiblePages % 2 === 0 ? 1 : 0));
      let endPage = Math.min(totalPages - 1, currentPage + halfVisible);
      
      // 调整 startPage 和 endPage 以确保显示 maxVisiblePages - 2 个中间页码
      const visibleCount = endPage - startPage + 1;
      if (visibleCount < maxVisiblePages - 2) {
        if (currentPage < halfVisible + 1) {
          endPage = Math.min(totalPages - 1, startPage + maxVisiblePages - 3);
        } else {
          startPage = Math.max(2, endPage - maxVisiblePages + 3);
        }
      }

      if (startPage > 2) {
        pageNumbers.push('...');
      }

      for (let i = startPage; i <= endPage; i++) {
        pageNumbers.push(i);
      }

      if (endPage < totalPages - 1) {
        pageNumbers.push('...');
      }

      pageNumbers.push(totalPages);
    }

    return pageNumbers.map((number, index) => (
      <button
        key={index}
        onClick={() => typeof number === 'number' && handlePageClick(number)}
        disabled={typeof number !== 'number' || number === currentPage}
        className={`mx-1 px-3 py-1 rounded text-sm ${
          number === currentPage
            ? 'bg-blue-600 text-white'
            : typeof number === 'number'
            ? 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-100'
            : 'text-gray-500 cursor-default'
        }`}
      >
        {number}
      </button>
    ));
  };

  return (
    <div className="flex justify-center items-center mt-8">
      <button
        onClick={() => handlePageClick(currentPage - 1)}
        disabled={currentPage === 1}
        className="mx-1 px-3 py-1 rounded text-sm bg-white text-gray-700 border border-gray-300 hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        上一页
      </button>
      
      {renderPageNumbers()}
      
      <button
        onClick={() => handlePageClick(currentPage + 1)}
        disabled={currentPage === totalPages}
        className="mx-1 px-3 py-1 rounded text-sm bg-white text-gray-700 border border-gray-300 hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        下一页
      </button>
    </div>
  );
};

export default Pagination; 