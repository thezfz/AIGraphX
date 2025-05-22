import React, { useMemo } from 'react';
import { useParams, Link } from 'react-router-dom';
import Graph from 'react-graph-vis';
import 'vis-network/styles/vis-network.css';
import { useModelRadialFocusGraphData } from '../services/apiQueries';
import type { components } from '../types/api';
import Spinner from '../components/common/Spinner';

type GraphData = components['schemas']['GraphData'];
type ApiNode = components['schemas']['Node'];
type ApiRelationship = components['schemas']['Relationship'];

const FocusGraphPage: React.FC = () => {
  const params = useParams<{ "*": string }>();
  const modelIdFromPath = params['*']; // Get modelId from splat route

  // Ensure modelId is a valid string before using it.
  const validModelId = modelIdFromPath || ""; 

  const {
    data: graphData,
    isLoading,
    isError,
    error,
  } = useModelRadialFocusGraphData(validModelId, !!validModelId); // Enable only if validModelId exists

  // Memoize graph object for react-graph-vis
  const graphVisData = useMemo(() => {
    if (!graphData || !graphData.nodes || !graphData.relationships) {
      return { nodes: [], edges: [] };
    }

    const nodes = graphData.nodes.map((node: ApiNode) => ({
      id: node.id,
      label: node.label || node.id,
      title: `Type: ${node.type}\nID: ${node.id}${node.label ? '\nLabel: ' + node.label : ''}`,
      color: node.id === validModelId ? '#FF4500' : (node.type === 'HFModel' ? '#FFD700' : (node.type === 'Paper' ? '#ADD8E6' : (node.type === 'Task' ? '#90EE90' : '#D3D3D3'))),
      shape: node.id === validModelId ? 'star' : 'dot',
      size: node.id === validModelId ? 30 : 15,
      fixed: node.id === validModelId ? { x: true, y: true } : false,
      x: node.id === validModelId ? 0 : undefined, // Position focus node at center
      y: node.id === validModelId ? 0 : undefined,
    }));

    const edges = graphData.relationships.map((rel: ApiRelationship) => ({
      id: `${rel.source}|${rel.target}|${rel.type}`,
      from: rel.source,
      to: rel.target,
      label: rel.type,
    }));
    return { nodes, edges };
  }, [graphData, validModelId]);

  const graphOptions = {
    layout: {
      hierarchical: false,
    },
    edges: {
      color: '#848484',
      arrows: {
        to: { enabled: true, scaleFactor: 0.5 },
      },
      smooth: {
        type: 'continuous',
      },
    },
    nodes: {
      borderWidth: 2,
    },
    physics: {
      enabled: true,
      forceAtlas2Based: {
        gravitationalConstant: -50, // Attracts nodes to center if centralGravity is high
        centralGravity: 0.015,      // Pulls nodes towards (0,0)
        springLength: 100,         // Preferred edge length
        springConstant: 0.08,
        avoidOverlap: 0.5,          // Try to prevent nodes from overlapping
      },
      solver: 'forceAtlas2Based', 
      stabilization: {
        iterations: 1000,
      },
    },
    interaction: {
      dragNodes: true,
      dragView: true,
      hover: true,
      zoomView: true,
      tooltipDelay: 200,
    },
    height: '800px', // Adjust as needed
  };

  if (!validModelId) {
    return (
      <div className="p-4 text-red-600">
        Error: Model ID is missing in the URL. Please provide a model ID, e.g., /focus-graph/your-model-id
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex justify-center items-center py-20">
        <Spinner />
        <p className="ml-2 text-gray-600">Loading focus graph for {validModelId}...</p>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="p-4 text-red-600">
        Error loading graph data for {validModelId}: {error?.message || 'Unknown error'}
      </div>
    );
  }

  if (!graphData || graphVisData.nodes.length === 0) {
    return (
      <div className="p-4 text-gray-600">
        No graph data found for model {validModelId}, or the model has no 1-hop connections based on the query.
      </div>
    );
  }

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">
        Focus Graph: <Link to={`/models/${encodeURIComponent(validModelId)}`} className="text-blue-600 hover:underline">{validModelId}</Link>
      </h1>
      <div className="border rounded-md shadow-lg">
        <Graph
          graph={graphVisData}
          options={graphOptions}
          // events={graphEvents} // Optional: add event handlers
          style={{ width: '100%', height: graphOptions.height }}
        />
      </div>
    </div>
  );
};

export default FocusGraphPage; 