import React, { useEffect, useRef, useState } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import * as THREE from 'three'; // 导入 Three.js
import { useAllGraphData } from '../services/apiQueries'; // Changed to useAllGraphData
// import { useParams } from 'react-router-dom'; // 如果需要从URL获取初始modelId (已注释)

// API返回的节点和关系类型 (与ModelDetailPage类似，可能需要调整)
import type { components } from '../types/api'; 
type ApiNode = components['schemas']['Node'];
type ApiRelationship = components['schemas']['Relationship'];

// 定义GraphData的期望结构，以便 ForceGraph3D 正确推断类型
interface MyNode {
  id: string | number;
  name: string;
  val: number;
  color: string;
  rawType?: string; 
  originalData?: ApiNode;
  // react-force-graph may add these
  x?: number;
  y?: number;
  z?: number;
  vx?: number;
  vy?: number;
  vz?: number;
}

interface MyLink {
  source: string | number | MyNode; // Can be ID or node object
  target: string | number | MyNode; // Can be ID or node object
  name?: string;
}

interface MyGraphData {
  nodes: MyNode[];
  links: MyLink[];
}

// const DEFAULT_FOCUS_MODEL_ID = 'deepseek-ai/DeepSeek-R1'; // No longer needed for global view

// Helper to assign colors based on node type
const getNodeColor = (nodeType: string | undefined): string => {
  switch (nodeType) {
    case 'HFModel': return 'rgba(255, 99, 132, 1)'; // Reddish pink, full opacity
    case 'Paper': return 'rgba(54, 162, 235, 1)';   // Blue, full opacity
    case 'Task': return 'rgba(75, 192, 192, 1)';    // Green, full opacity
    case 'Dataset': return 'rgba(255, 206, 86, 1)';  // Yellow, full opacity
    case 'Method': return 'rgba(153, 102, 255, 1)'; // Purple, full opacity
    default: return 'rgba(201, 203, 207, 1)';     // Grey, full opacity
  }
};

const NODE_TYPES_FOR_LEGEND = ['HFModel', 'Paper', 'Task', 'Dataset', 'Method', 'Other'];

// 创建可重用的几何体以提高性能
const nodeGeometry = new THREE.SphereGeometry(1, 12, 6); // Further reduced segments

// 自定义节点对象的函数
const renderCustomNodeObject = (node: MyNode): THREE.Object3D => {
  const material = new THREE.MeshPhongMaterial({ // 使用MeshPhongMaterial以获得光泽效果
    color: node.color, 
    shininess: 90,     // 增加光泽度，使高光更锐利
    transparent: true, 
    opacity: node.color.includes('rgba') ? parseFloat(node.color.split(',')[3]) || 1 : 1 
  });

  const mesh = new THREE.Mesh(nodeGeometry, material);
  
  // 根据 node.val 调整大小，使用立方根来缓和大小差异，再乘以一个系数
  // 基础半径为1（来自nodeGeometry），所以scale直接作用于半径
  const scale = Math.cbrt(node.val || 1) * 2.0; // 调整乘数以获得合适的基础大小
  mesh.scale.set(scale, scale, scale);

  return mesh;
};

const GlobalGraphPage: React.FC = () => {
  const graphRef = useRef<any>(); // For accessing graph instance methods
  // const { modelIdFromUrl } = useParams(); // Example if you want to use URL param
  const [graphData, setGraphData] = useState<MyGraphData>({ nodes: [], links: [] });
  const addedLightsRef = useRef<THREE.PointLight[]>([]); // Ref to keep track of added lights

  const {
    data: apiGraphData, // This is { nodes: ApiNode[], relationships: ApiRelationship[] }
    isLoading,
    isError,
    error,
    // refetch // refetch might not be needed if we load all data at once and don't change focus
  } = useAllGraphData(); // Use the new hook for fetching all graph data

  useEffect(() => {
    if (apiGraphData && apiGraphData.nodes && apiGraphData.relationships) {
      const nodes: MyNode[] = apiGraphData.nodes.map((node: ApiNode) => ({
        id: node.id!,
        name: node.label || node.id!,
        val: node.type === 'HFModel' ? 10 : (node.type === 'Paper' ? 5 : 2), // Larger HFModels
        color: getNodeColor(node.type),
        rawType: node.type,
        originalData: node,
      }));

      const links: MyLink[] = apiGraphData.relationships.map((rel: ApiRelationship) => ({
        source: rel.source!,
        target: rel.target!,
        name: rel.type, // e.g., "DERIVED_FROM", "CITES"
      })); 
      
      const uniqueNodes = Array.from(new Map(nodes.map(n => [n.id, n])).values());
      // Ensure links only connect nodes present in uniqueNodes for graph integrity
      const nodeIds = new Set(uniqueNodes.map(n => n.id));
      const validLinks = links.filter(l => nodeIds.has(l.source as string | number) && nodeIds.has(l.target as string | number));
      const uniqueLinks = Array.from(new Map(validLinks.map(l => [`${l.source}-${l.target}-${l.name}`, l])).values());

      setGraphData({ nodes: uniqueNodes, links: uniqueLinks });
    }
  }, [apiGraphData]);
  
  // Auto-rotate camera and initial position adjustment, and custom lights
  useEffect(() => {
    if (graphRef.current && graphData.nodes.length > 0) { 
        graphRef.current.cameraPosition({ x:0, y:0, z: 350 }, undefined, 1000); 

        if (graphRef.current.controls) { 
            const controls = graphRef.current.controls();
            controls.autoRotate = true;
            controls.autoRotateSpeed = 0.3;
        }

        // Add custom lights if not already added
        if (addedLightsRef.current.length === 0) {
          const scene = graphRef.current.scene();
          
          const light1 = new THREE.PointLight(0xff00ff, 0.6, 0); // Magenta, intensity, distance (0=infinite)
          light1.position.set(200, 150, 250);
          scene.add(light1);
          addedLightsRef.current.push(light1);

          const light2 = new THREE.PointLight(0x00ffff, 0.6, 0); // Cyan
          light2.position.set(-200, -150, 250);
          scene.add(light2);
          addedLightsRef.current.push(light2);
        }
    }
    return () => {
        if(graphRef.current && graphRef.current.controls){
            const controls = graphRef.current.controls();
            if (controls) {
                controls.autoRotate = false;
            }
        }
        // Cleanup custom lights
        if (graphRef.current && addedLightsRef.current.length > 0) {
          const scene = graphRef.current.scene();
          addedLightsRef.current.forEach(light => scene.remove(light));
          addedLightsRef.current = [];
        }
    };
  }, [graphData]); // Re-run when graphData changes


  const handleNodeClick = (node: MyNode) => {
    console.log('Clicked node:', node);
    // Simplified: just focus camera on clicked node for now
    // Removed logic for setting currentFocusModelId and refetching data
    if (graphRef.current) {
      if (node.x !== undefined && node.y !== undefined && node.z !== undefined) {
        graphRef.current.centerAt(node.x, node.y, 1000); 
        graphRef.current.cameraPosition({ x: node.x + 60, y: node.y + 60, z: node.z + 60 }, undefined, 1000);
      }
    }
  };

  if (isLoading) return <div className="p-4 text-center">加载全局图谱数据中...</div>; // Updated loading text
  if (isError) return <div className="p-4 text-center text-red-500">加载全局图谱数据失败: {error?.message || '未知错误'}</div>;

  return (
    <div className="p-4">
      <h2 className="text-xl font-semibold mb-2">全局知识图谱 (3D)</h2> {/* Updated title */}
      <div className="mb-2 text-sm">
        提示: 探索完整的知识图谱。悬停查看节点和链接详情。
      </div>
      <div className="mb-2 text-xs text-gray-500">
        交互提示: 左键拖拽背景旋转视角，鼠标滚轮缩放，右键拖拽背景平移视图。按住Shift并左键拖拽节点可移动节点。
      </div>
      <div style={{ position: 'relative', width: '100%', height: 'calc(80vh - 70px)', border: '1px solid #eee', background: '#111827' }}> {/* Dark background */}
        <div className="absolute top-2 left-2 bg-gray-700 bg-opacity-60 p-2 rounded text-white text-xs z-10">
          <h4 className="font-bold mb-1">图例:</h4>
          {NODE_TYPES_FOR_LEGEND.map(type => (
            <div key={type} className="flex items-center mb-0.5">
              <span style={{ backgroundColor: getNodeColor(type === 'Other' ? undefined : type), width: '12px', height: '12px', marginRight: '5px', display: 'inline-block', border: '1px solid #ccc' }}></span>
              {type}
            </div>
          ))}
        </div>
        {graphData.nodes.length > 0 ? (
            <ForceGraph3D
              ref={graphRef}
              graphData={graphData}
              nodeThreeObject={renderCustomNodeObject} // 使用自定义节点渲染
              nodeLabel={(node: MyNode) => `${node.name} (${node.rawType || 'Unknown Type'})`}
              linkColor={(link: MyLink) => link.name === 'DERIVED_FROM' ? 'rgba(144, 238, 144, 0.9)' : 'rgba(255,255,255,0.3)'} // Light green for DERIVED_FROM
              linkWidth={(link: MyLink) => link.name === 'DERIVED_FROM' ? 1.2 : 0.3}
              linkLabel={(link: MyLink) => link.name || ''}
              linkDirectionalParticles={0} // Disabled particles for performance
              linkDirectionalParticleWidth={(link: MyLink) => link.name === 'DERIVED_FROM' ? 2 : 1}
              linkDirectionalParticleSpeed={0.006}
              linkHoverPrecision={8}
              onNodeClick={handleNodeClick as any}
              cooldownTime={20000} // Stop engine after 20 seconds of inactivity
              onEngineStop={() => {
                console.log('Force graph engine stopped.');
                // Optionally, explicitly pause rendering loop if needed, though cooldownTime should suffice
                // if (graphRef.current) graphRef.current.pauseAnimation(); 
              }}
            />
        ) : (
          <p className="text-center text-gray-500 pt-10">未能加载图谱数据或图中无节点。</p>
        )}
      </div>
    </div>
  );
};

export default GlobalGraphPage; 