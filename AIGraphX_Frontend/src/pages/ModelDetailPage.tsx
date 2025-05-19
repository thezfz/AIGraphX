import React, { useEffect, useState, useMemo } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useModelDetail, useModelGraphData } from '../services/apiQueries'; // 修正导入路径
import Spinner from '../components/common/Spinner'; // 引入 Spinner
import { ArrowDownTrayIcon, HeartIcon } from '@heroicons/react/24/outline'; // 引入图标
import ReactMarkdown from 'react-markdown'; // ADDED
import remarkGfm from 'remark-gfm'; // ADDED

// 导入 react-graph-vis 和其 CSS
import Graph from 'react-graph-vis';
import 'vis-network/styles/vis-network.css';

// 修改导入以正确指向 components.schemas 下的类型
import type { components } from '../types/api'; 
type GraphData = components['schemas']['GraphData'];
type ApiNode = components['schemas']['Node'];
type ApiRelationship = components['schemas']['Relationship']; // 定义 Relationship 类型
// 如果还需要 Relationship 类型，可以类似添加：
// type ApiRelationship = components['schemas']['Relationship'];

const ModelDetailPage: React.FC = () => {
  const { modelId } = useParams<{ modelId: string }>();
  const [showDebugInfo, setShowDebugInfo] = useState(false); // 添加状态控制调试信息显示
  const [parentModels, setParentModels] = useState<ApiNode[]>([]);
  const [derivedModels, setDerivedModels] = useState<ApiNode[]>([]);

  // 获取模型文本详情 (假设这个hook仍然需要)
  const {
    data: modelDetails, // 重命名以区分图数据
    isLoading: isLoadingDetails,
    isError: isErrorDetails,
    error: errorDetails,
  } = useModelDetail(modelId); // 确保 modelId 非空再调用

  // 获取模型图数据
  const {
    data: modelGraphData,
    isLoading: isLoadingGraph,
    isError: isErrorGraph,
    error: errorGraph,
  } = useModelGraphData(modelId!, !!modelId); // 使用 modelId! 断言，因为前面有检查

  // 添加调试日志
  useEffect(() => {
    if (modelDetails) {
      console.log('Model TEXT details received:', modelDetails);
    }
    if (modelGraphData) {
      console.log('Model GRAPH data received:', modelGraphData);

      // 解析父模型和派生模型
      if (modelGraphData.nodes && modelGraphData.relationships && modelId) {
        const parents: ApiNode[] = [];
        const derived: ApiNode[] = [];
        const currentModelNodeId = modelId; // 使用从 useParams 获取的 modelId 作为中心
        console.log(`[Debug DERIVED_FROM] Processing relationships for currentModelNodeId: ${currentModelNodeId}`);

        modelGraphData.relationships.forEach(rel => {
          console.log(`[Debug DERIVED_FROM] Inspecting relationship: source=${rel.source}, target=${rel.target}, type=${rel.type}`);
          if (rel.type === 'DERIVED_FROM') {
            console.log(`[Debug DERIVED_FROM] Found DERIVED_FROM: source=${rel.source}, target=${rel.target}`);
            // 如果当前模型是目标 (B in A->B), 意味着 A (source) 是从 B (target/currentModel) 派生出来的。
            // 所以 A (rel.source) 是派生模型。
            if (rel.target === currentModelNodeId) {
              console.log(`[Debug DERIVED_FROM] Match for Derived: ${rel.source} -> ${currentModelNodeId}`);
              const derivedNode = modelGraphData.nodes.find(n => n.id === rel.source);
              if (derivedNode) {
                derived.push(derivedNode);
                console.log(`[Debug DERIVED_FROM] Added Derived Node:`, derivedNode);
              } else {
                console.warn(`[Debug DERIVED_FROM] Derived node with id ${rel.source} not found in modelGraphData.nodes`);
              }
            }
            // 如果当前模型是源 (A in A->B), 意味着 A (source/currentModel) 是从 B (target) 派生出来的。
            // 所以 B (rel.target) 是父模型。
            else if (rel.source === currentModelNodeId) {
              console.log(`[Debug DERIVED_FROM] Match for Parent: ${currentModelNodeId} -> ${rel.target}`);
              const parentNode = modelGraphData.nodes.find(n => n.id === rel.target);
              if (parentNode) {
                parents.push(parentNode);
                console.log(`[Debug DERIVED_FROM] Added Parent Node:`, parentNode);
              } else {
                console.warn(`[Debug DERIVED_FROM] Parent node with id ${rel.target} not found in modelGraphData.nodes`);
              }
            }
          }
        });
        setParentModels(parents);
        setDerivedModels(derived);
        console.log('Parent models:', parents);
        console.log('Derived models:', derived);
      }

      // 主动检测重复NODE ID
      if (modelGraphData.nodes) {
        const nodeIds = modelGraphData.nodes.map(n => n.id);
        const uniqueNodeIds = new Set(nodeIds);
        if (nodeIds.length !== uniqueNodeIds.size) {
            console.error('DUPLICATE NODE IDs DETECTED IN modelGraphData.nodes:', modelGraphData.nodes);
            const idCounts: Record<string, number> = {};
            nodeIds.forEach(id => {
                idCounts[id] = (idCounts[id] || 0) + 1;
            });
            const duplicates = Object.entries(idCounts).filter(([/*id*/, count]) => count > 1);
            console.error('Duplicate IDs and their counts:', duplicates);
        } else {
            console.log('No duplicate node IDs found in modelGraphData.nodes.');
        }
      }

      // 主动检测重复 RELATIONSHIP
      if (modelGraphData.relationships) {
        const relSignatures = modelGraphData.relationships.map(
            r => `${r.source}|${r.target}|${r.type}`
        );
        const uniqueRelSignatures = new Set(relSignatures);
        if (relSignatures.length !== uniqueRelSignatures.size) {
            console.error(
                'DUPLICATE RELATIONSHIPS DETECTED in modelGraphData.relationships:',
                modelGraphData.relationships
            );
            const relCounts: Record<string, number> = {};
            relSignatures.forEach(sig => {
                relCounts[sig] = (relCounts[sig] || 0) + 1;
            });
            const duplicateRels = Object.entries(relCounts).filter(
                ([/*sig*/, count]) => count > 1
            );
            console.error('Duplicate relationship signatures and their counts:', duplicateRels);
        } else {
            console.log('No duplicate relationships found in modelGraphData.relationships.');
        }
      }
    }
  }, [modelDetails, modelGraphData, modelId]);

  // --- 数据转换为图表库所需的格式 ---
  const graph = useMemo(() => {
    if (!modelGraphData || !modelGraphData.nodes || !modelGraphData.relationships) {
      return { nodes: [], edges: [] };
    }
    const currentModelNodeId = modelId;
    const nodes = modelGraphData.nodes.map(node => {
      let color = '#CBD5E1'; // 默认浅灰蓝
      let shape = 'dot';
      let borderWidth = 2;
      let size = 16;
      let fontColor = '#F3F4F6'; // 浅色字体
      if (node.type === 'HFModel') {
        if (node.id === currentModelNodeId) {
          color = '#38BDF8'; // 天蓝
          shape = 'diamond';
          size = 32;
          borderWidth = 4;
        } else if (parentModels.some(p => p.id === node.id)) {
          color = '#F472B6'; // 粉色
          shape = 'triangle';
          size = 22;
        } else if (derivedModels.some(d => d.id === node.id)) {
          color = '#A7F3D0'; // 浅绿色
          shape = 'triangleDown';
          size = 22;
        } else {
          color = '#FBBF24'; // 金色
          shape = 'dot';
        }
      } else if (node.type === 'Paper') {
        color = '#818CF8'; // 紫色
        shape = 'box';
      } else if (node.type === 'Task') {
        color = '#FDE68A'; // 浅黄
        shape = 'square';
      } else if (node.type === 'Dataset') {
        color = '#6EE7B7'; // 青绿色
        shape = 'database';
      } else if (node.type === 'Method') {
        color = '#FCA5A5'; // 浅红
        shape = 'hexagon';
      }
      return {
        id: node.id,
        label: node.label || node.id,
        title: `Type: ${node.type}\nID: ${node.id}${node.label ? '\nLabel: ' + node.label : ''}`,
        color: color,
        shape: shape,
        size: size,
        font: { size: 13, color: fontColor },
        borderWidth: borderWidth,
      };
    });
    const edges = modelGraphData.relationships.map((rel: ApiRelationship) => ({
      id: `${rel.source}|${rel.target}|${rel.type}`,
      from: rel.source,
      to: rel.target,
      label: rel.type,
      color: { color: '#E5E7EB', opacity: 0.7 }, // 浅灰
      font: { size: 10, color: '#F3F4F6', background: 'rgba(30,41,59,0.7)', strokeWidth:0 },
      dashes: rel.type === 'DERIVED_FROM', // DERIVED_FROM 用虚线
      smooth: { type: 'dynamic' }, // 曲线
    }));
    return { nodes, edges };
  }, [modelGraphData, modelId, parentModels, derivedModels]);

  // --- 图表选项 ---
  const graphOptions = {
    layout: {
      hierarchical: false,
    },
    edges: {
      color: '#E5E7EB',
      arrows: {
        to: { enabled: true, scaleFactor: 0.5 }
      },
      smooth: {
        type: 'dynamic',
      },
      width: 1.2,
      shadow: true,
    },
    nodes: {
      shape: 'dot',
      size: 16,
      font: {
        size: 13,
        color: '#F3F4F6',
      },
      borderWidth: 2,
      shadow: true,
    },
    physics: {
      enabled: true,
      forceAtlas2Based: {
        gravitationalConstant: -60,
        centralGravity: 0.02,
        springLength: 120,
        springConstant: 0.09,
        avoidOverlap: 1.2,
      },
      solver: 'forceAtlas2Based',
      stabilization: {
        iterations: 1200,
      }
    },
    interaction: {
      dragNodes: true,
      dragView: true,
      hover: true,
      zoomView: true,
      tooltipDelay: 200,
    },
    height: '600px',
  };

  // --- 事件处理 (示例) ---
  const graphEvents = {
    selectNode: ({ nodes }: { nodes: string[] }) => {
      if (nodes.length > 0) {
        console.log('Selected node IDs:', nodes);
        // 可以根据选中的节点ID做进一步操作，比如显示更详细的信息或导航
      }
    },
    // selectEdge: ({ edges }: { edges: string[] }) => {
    //   console.log('Selected edge IDs:', edges);
    // },
  };

  // 如果没有 modelId，直接显示错误信息
  if (!modelId) {
    return (
      <div className="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded relative" role="alert">
        <strong className="font-bold">无效请求: </strong>
        <span className="block sm:inline">未提供模型 ID。</span>
      </div>
    );
  }

  // 合并加载状态
  if (isLoadingDetails || (!!modelId && isLoadingGraph)) { // 只有在 modelId 存在时才考虑 isLoadingGraph
    return (
      <div className="flex justify-center items-center py-20">
        <Spinner /> {/* 使用 Spinner 组件 */}
        <p className="ml-2 text-gray-600">正在加载模型 {modelId} 的详情与图数据...</p>
      </div>
    );
  }

  // 合并错误状态
  if (isErrorDetails) {
    return (
      <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
        <strong className="font-bold">获取模型详情出错: </strong>
        <span className="block sm:inline">{errorDetails instanceof Error ? errorDetails.message : JSON.stringify(errorDetails)}</span>
        <div className="mt-2">
          <p className="text-sm">请求的模型 ID: {modelId}</p>
          <p className="text-sm">您可以尝试刷新页面或返回搜索。</p>
        </div>
      </div>
    );
  }
  // 图数据错误优先于文本详情未找到，因为图是额外信息
  if (isErrorGraph && modelId) { // 只在尝试加载图数据时显示图错误
     console.warn("Error loading graph data, but showing text details.", errorGraph);
     // 不阻塞页面，允许显示文本详情，可以在图表区域显示错误信息
  }

  if (!modelDetails) {
    return (
      <div className="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded relative" role="alert">
        <strong className="font-bold">未找到: </strong>
        <span className="block sm:inline">无法找到 ID 为 {modelId} 的模型文本详情。</span>
        <div className="mt-2">
          <p className="text-sm">可能的原因：</p>
          <ul className="list-disc list-inside text-sm">
            <li>模型 ID 不存在</li>
            <li>数据库中没有该模型的信息</li>
            <li>服务器响应格式与预期不符</li>
          </ul>
        </div>
      </div>
    );
  }

  // 安全获取属性的辅助函数
  const safeRender = (render: () => React.ReactNode) => {
    try {
      return render();
    } catch (error) {
      console.error('Render error:', error);
      return <span className="text-red-500">渲染错误</span>;
    }
  };

  // Preprocess readme_content to remove HTML-like comments
  const cleanReadmeContent = modelDetails?.readme_content?.replace(/<!--.*?-->/gs, '') || "";

  // --- 渲染从 API 获取的模型数据 ---
  // 注意：这里的字段需要匹配 useModelDetail 返回的 ModelDetailResponse 类型
  try {
    return (
      <div className="bg-white rounded-lg shadow-md overflow-hidden"> {/* 添加 overflow-hidden 防止内容溢出 */} 
        {/* Header Section */}
        <div className="p-6 bg-gradient-to-r from-blue-50 to-indigo-50"> {/* 添加渐变背景 */} 
          <h1 className="text-2xl md:text-3xl font-bold text-gray-800 mb-1 break-all">
            {safeRender(() => modelDetails.model_id)}
          </h1>
          {modelDetails.author && (
            <p className="text-md text-gray-600">由 {safeRender(() => modelDetails.author)} 创建</p>
          )}
        </div>

        {/* Main Content Section */}
        <div className="p-6">
          {/* Metadata and Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            {/* Left Column: Metadata */}
            <div className="md:col-span-2 space-y-4"> {/* 使用 space-y 控制间距 */}
              {modelDetails.pipeline_tag && (
                <div>
                  <h3 className="text-sm font-medium text-gray-500">任务类型</h3>
                  <p className="text-lg text-gray-800">{safeRender(() => modelDetails.pipeline_tag)}</p>
                </div>
              )}
              {modelDetails.library_name && (
                <div>
                  <h3 className="text-sm font-medium text-gray-500">库</h3>
                  <p className="text-lg text-gray-800">{safeRender(() => modelDetails.library_name)}</p>
                </div>
              )}
              {modelDetails.last_modified && (
                <div>
                  <h3 className="text-sm font-medium text-gray-500">最后更新</h3>
                  <p className="text-lg text-gray-800">
                    {safeRender(() => modelDetails.last_modified ? new Date(modelDetails.last_modified).toLocaleDateString('zh-CN', { year: 'numeric', month: 'long', day: 'numeric' }) : '未知')} {/* 改进日期格式 */} 
                  </p>
                </div>
              )}

              {/* Parent Models List */}
              {parentModels.length > 0 && (
                <div className="mt-4">
                  <h3 className="text-sm font-medium text-gray-500">父模型 (Base Models)</h3>
                  <ul className="list-disc list-inside pl-4 space-y-1">
                    {parentModels.map(node => (
                      <li key={node.id} className="text-gray-700">
                        <Link to={`/models/${encodeURIComponent(node.id)}`} className="text-blue-600 hover:text-blue-800 hover:underline">
                          {node.label || node.id}
                        </Link>
                        {node.type && <span className="text-xs text-gray-500 ml-2">({node.type})</span>}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Derived Models List */}
              {derivedModels.length > 0 && (
                <div className="mt-4">
                  <h3 className="text-sm font-medium text-gray-500">派生模型 (Derived Models)</h3>
                  <ul className="list-disc list-inside pl-4 space-y-1">
                    {derivedModels.map(node => (
                      <li key={node.id} className="text-gray-700">
                        <Link to={`/models/${encodeURIComponent(node.id)}`} className="text-blue-600 hover:text-blue-800 hover:underline">
                          {node.label || node.id}
                        </Link>
                        {node.type && <span className="text-xs text-gray-500 ml-2">({node.type})</span>}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            {/* Right Column: Stats */}
            <div className="bg-gray-50 rounded-lg p-4 space-y-3"> {/* 使用 space-y */} 
              <h2 className="text-lg font-semibold text-gray-700 mb-2">统计信息</h2>
              {modelDetails.downloads !== undefined && modelDetails.downloads !== null && (
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600 inline-flex items-center">
                    <ArrowDownTrayIcon className="mr-1.5 h-5 w-5 text-gray-400" aria-hidden="true" />
                    下载次数
                  </span>
                  <span className="font-medium text-gray-800">
                    {safeRender(() => modelDetails.downloads?.toLocaleString() || '0')}
                  </span>
                </div>
              )}
              {modelDetails.likes !== undefined && modelDetails.likes !== null && (
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600 inline-flex items-center">
                    <HeartIcon className="mr-1.5 h-5 w-5 text-gray-400" aria-hidden="true" />
                    点赞数
                  </span>
                  <span className="font-medium text-gray-800">
                    {safeRender(() => modelDetails.likes?.toLocaleString() || '0')}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Tags Section */}
          {modelDetails.tags && Array.isArray(modelDetails.tags) && modelDetails.tags.length > 0 && (
            <div className="mb-6">
              <h3 className="text-md font-semibold text-gray-700 mb-2">标签</h3>
              <div className="flex flex-wrap gap-2">
                {modelDetails.tags.map((tag: string, index: number) => (
                  <span key={index} className="bg-indigo-100 text-indigo-800 text-xs font-medium px-3 py-1 rounded-full">
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* --- Knowledge Graph Section --- */}
          <div className="mt-6 pt-6 border-t border-gray-200 bg-gray-800 rounded-lg shadow-lg p-4"> {/* 深色背景+圆角+阴影 */}
            <h3 className="text-lg font-semibold text-gray-100 mb-3">知识图谱关联</h3>
            {isLoadingGraph && !isErrorGraph && (
              <div className="flex items-center text-gray-300">
                <Spinner />
                <span className="ml-2">正在加载图谱数据...</span>
              </div>
            )}
            {isErrorGraph && (
              <div className="bg-red-50 border border-red-300 text-red-700 px-3 py-2 rounded text-sm">
                加载图谱数据失败: {errorGraph instanceof Error ? errorGraph.message : JSON.stringify(errorGraph)}
              </div>
            )}
            {!isLoadingGraph && !isErrorGraph && modelGraphData && graph.nodes.length > 0 && (
              <div className="border rounded-md overflow-hidden bg-gray-900"> {/* 图区域更深色 */}
                <Graph
                  key={modelId}
                  graph={graph}
                  options={graphOptions}
                  events={graphEvents}
                  style={{ width: '100%', height: graphOptions.height, background: '#1a202c' }} // 深色背景
                />
              </div>
            )}
            {!isLoadingGraph && !isErrorGraph && (!modelGraphData || graph.nodes.length === 0) && (
              <p className="text-gray-400 text-sm">未找到该模型的图谱关联数据。</p>
            )}
          </div>

          {/* ADDED: Readme Content Section */}
          {modelDetails.readme_content && (
            <div className="mt-6 pt-6 border-t border-gray-200">
              <h3 className="text-lg font-semibold text-gray-700 mb-3">README</h3>
              <div className="bg-gray-50 p-4 rounded-md overflow-x-auto">
                <div className="prose prose-base max-w-none text-gray-700">
                  <ReactMarkdown 
                    remarkPlugins={[remarkGfm]}
                    components={{
                      table: ({node, ...props}) => <table className="w-full text-sm border-collapse border border-gray-300" {...props} />,
                      thead: ({node, ...props}) => <thead className="bg-gray-100" {...props} />,
                      th: ({node, ...props}) => <th className="p-2 border border-gray-300 !text-center font-semibold" {...props} />,
                      td: ({node, ...props}) => <td className="p-2 border border-gray-300 !text-center" {...props} />,
                      // You can add more custom renderers for tr, tbody, etc. if needed
                    }}
                  >
                    {cleanReadmeContent}
                  </ReactMarkdown>
                </div>
              </div>
            </div>
          )}

          {/* ADDED: Dataset Links Section */}
          {modelDetails.dataset_links && Array.isArray(modelDetails.dataset_links) && modelDetails.dataset_links.length > 0 && (
            <div className="mt-6 pt-6 border-t border-gray-200">
              <h3 className="text-lg font-semibold text-gray-700 mb-3">相关数据集</h3>
              <ul className="list-disc list-inside space-y-1 pl-2">
                {modelDetails.dataset_links.map((link: string, index: number) => (
                  <li key={index} className="text-sm">
                    <a 
                      href={link} 
                      target="_blank" 
                      rel="noopener noreferrer" 
                      className="text-blue-600 hover:text-blue-800 hover:underline break-all"
                      title={link}
                    >
                      {link}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Debug Info Section (Toggleable) */}
          {import.meta.env.DEV && (
            <div className="mt-8 border-t pt-4">
              <button
                onClick={() => setShowDebugInfo(!showDebugInfo)}
                className="text-sm text-blue-600 hover:text-blue-800 mb-2"
              >
                {showDebugInfo ? '隐藏' : '显示'}调试信息
              </button>
              {showDebugInfo && (
                <div className="p-4 bg-gray-100 rounded-lg overflow-x-auto">
                  <pre className="text-xs">{JSON.stringify(modelDetails, null, 2)}</pre>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  } catch (renderError) {
    console.error('Fatal render error:', renderError);
    return (
      <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
        <strong className="font-bold">渲染错误: </strong>
        <span className="block sm:inline">渲染模型详情页面时发生错误。</span>
        <div className="mt-4 p-2 bg-white rounded overflow-x-auto">
          <pre className="text-xs">{JSON.stringify({ error: String(renderError), modelId, modelKeys: modelDetails ? Object.keys(modelDetails) : [] }, null, 2)}</pre>
        </div>
      </div>
    );
  }
};

export default ModelDetailPage; 