import faiss  # type: ignore[import-untyped]
import numpy as np
import json
import os

# --- 配置 ---
DIMENSION = 8
NUM_VECTORS = 4
TEST_DATA_DIR = "tests/test_data"  # 将测试数据放在 tests/test_data/
INDEX_FILENAME = "test_papers.index"
ID_MAP_FILENAME = "test_papers_ids.json"

# 确保目录存在
os.makedirs(TEST_DATA_DIR, exist_ok=True)

INDEX_PATH = os.path.join(TEST_DATA_DIR, INDEX_FILENAME)
ID_MAP_PATH = os.path.join(TEST_DATA_DIR, ID_MAP_FILENAME)

# --- 生成向量 ---
# 使用易于区分的向量
vectors = np.array(
    [
        [0.1] * DIMENSION,
        [0.9] * DIMENSION,
        [0.5] * DIMENSION,
        [-0.5] * DIMENSION,  # 包含负值示例
    ]
).astype("float32")

print(f"Generated vectors with shape: {vectors.shape}")

# L2 归一化 (根据实际嵌入是否归一化决定是否取消注释)
# print("Applying L2 normalization...")
# faiss.normalize_L2(vectors)
# print("Vectors after normalization:\\n", vectors)

# --- 生成 ID 映射 ---
# 使用非连续的整数 ID
id_map = {
    "0": 101,  # Faiss index 0 maps to paper_id 101
    "1": 256,  # Faiss index 1 maps to paper_id 256
    "2": 999,  # Faiss index 2 maps to paper_id 999
    "3": 42,  # Faiss index 3 maps to paper_id 42
}
print(f"Generated ID map: {id_map}")

# --- 创建并保存 Faiss 索引 ---
# 使用简单的 IndexFlatL2 (欧氏距离)
index = faiss.IndexFlatL2(DIMENSION)
# index = faiss.IndexFlatIP(DIMENSION) # 如果使用内积(余弦相似度需要归一化向量)

print(f"Index is trained: {index.is_trained}")  # 对于 IndexFlat 总是 True
index.add(vectors)
print(f"Number of vectors in index: {index.ntotal}")

print(f"Writing index to {INDEX_PATH}")
faiss.write_index(index, INDEX_PATH)

# --- 保存 ID 映射 ---
print(f"Writing ID map to {ID_MAP_PATH}")
with open(ID_MAP_PATH, "w") as f:
    json.dump(id_map, f, indent=2)

print("\nTest Faiss index and ID map generated successfully.")

# --- 可选：测试加载和搜索 (验证生成是否正确) ---
try:
    print("\n--- Verifying generated files ---")
    loaded_index = faiss.read_index(INDEX_PATH)
    print(f"Loaded index ntotal: {loaded_index.ntotal}")
    with open(ID_MAP_PATH, "r") as f:
        loaded_id_map_str_keys = json.load(f)
        # JSON keys are strings, need conversion if used directly as int keys
        loaded_id_map = {int(k): v for k, v in loaded_id_map_str_keys.items()}
    print(f"Loaded ID map (keys as int): {loaded_id_map}")

    # 构造一个查询向量，应该最接近第一个向量 ([0.1]*8)
    query_vector = np.array([[0.11] * DIMENSION]).astype("float32")
    # faiss.normalize_L2(query_vector) # 如果向量已归一化，查询向量也应归一化

    k = 3
    distances, indices = loaded_index.search(query_vector, k)
    print(f"\nSearch results for query close to vector 0 (k={k}):")
    print(f"Distances: {distances}")  # L2 距离，越小越近
    print(f"Indices: {indices}")  # Faiss 内部索引位置 (0, 1, 2...)

    # 将内部索引映射回 paper_id
    results = []
    if len(indices) > 0:
        for i, internal_idx in enumerate(indices[0]):
            if internal_idx != -1:  # -1 表示没有找到足够的邻居
                paper_id = loaded_id_map.get(internal_idx)  # Use int key
                if paper_id is not None:
                    results.append((paper_id, float(distances[0][i])))
    print(f"Mapped results (paper_id, distance): {results}")
    # 预期 paper_id 101 应该是第一个 (距离最小)
    if results and results[0][0] == 101:
        print("Verification successful: Closest vector found correctly.")
    else:
        print("Verification failed: Closest vector mismatch.")

except Exception as e:
    print(f"\nError during verification: {e}")
