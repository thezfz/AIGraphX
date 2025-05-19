import React from 'react';

const GlobalGraphPage: React.FC = () => {
  return (
    <div className="p-4">
      <h2 className="text-2xl font-semibold mb-4">全局模型关系图 (3D)</h2>
      <div id="3d-graph" style={{ width: '100%', height: '70vh', border: '1px solid #ccc' }}>
        {/* 3D Graph will be rendered here */}
        <p className="text-center text-gray-500 pt-10">3D 图谱加载中或即将显示...</p>
      </div>
    </div>
  );
};

export default GlobalGraphPage; 