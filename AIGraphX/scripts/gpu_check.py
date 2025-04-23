#!/usr/bin/env python
# -*- coding: utf-8 -*- # 指定编码为 UTF-8

# 文件作用说明：
# 该脚本用于检查当前环境中 PyTorch 和 Faiss 是否能够正确检测和使用 GPU (特别是 CUDA)。
# 这对于需要 GPU 加速的机器学习任务（如文本嵌入、大规模相似性搜索）非常重要。
# 脚本会依次执行以下操作：
# 1. 使用 PyTorch 检查 CUDA 是否可用、GPU 数量以及 GPU 名称。
# 2. 使用 Faiss 检查可用的 GPU 数量。
# 3. 如果 PyTorch 和 Faiss 都检测到 GPU，则尝试创建一个简单的 Faiss GPU 索引，
#    添加一些随机数据，并执行一次搜索，以验证 Faiss GPU 功能是否基本可用。
# 4. 打印检查结果和任何遇到的错误。
#
# 运行方式：
# 直接在安装了 PyTorch (带 CUDA 支持) 和 Faiss (带 GPU 支持) 的环境中执行：
# python scripts/gpu_check.py
#
# 交互对象：
# - 依赖：PyTorch 库, Faiss 库, NumPy 库。
# - 输出：打印 GPU 检测结果和 Faiss GPU 功能测试结果到控制台。

import torch # 导入 PyTorch 库
import faiss # type: ignore[import-untyped] # 导入 Faiss 库, 忽略 Faiss 的类型检查
#              # type: ignore[import-untyped] 忽略 Faiss 的类型检查
import numpy as np # 导入 NumPy 库，用于创建测试数据

print("--- PyTorch GPU 检查 ---")
# 检查 PyTorch 是否能够找到并使用 CUDA 加速的 GPU
# 这是使用 GPU 的先决条件
is_available = torch.cuda.is_available()
print(f"torch.cuda.is_available(): {is_available}")

# 如果 PyTorch 检测到 CUDA 可用
if is_available:
    # 获取可用的 GPU 数量
    device_count = torch.cuda.device_count()
    print(f"可用的 GPU 数量: {device_count}")
    # 遍历每个可用的 GPU 并打印其名称
    for i in range(device_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    # 如果 PyTorch 无法检测到 CUDA
    print("PyTorch 未检测到 CUDA。Faiss GPU 功能很可能无法使用。")

print("\n--- Faiss GPU 检查 ---")
try:
    # 尝试获取 Faiss 能识别的 GPU 数量
    # 注意：Faiss 对 GPU 的检测独立于 PyTorch，但通常依赖相同的 CUDA 环境
    ngpu = faiss.get_num_gpus()
    print(f"Faiss 检测到 {ngpu} 个 GPU。")

    # 只有当 Faiss 检测到 GPU 并且 PyTorch 也认为 CUDA 可用时，才进行 Faiss GPU 功能测试
    if ngpu > 0 and is_available:
        print("尝试在 GPU 上创建并测试 Faiss 索引...")
        # --- 创建一个简单的 Faiss GPU 测试 ---
        d = 64      # 定义向量维度 (dimension)
        nb = 1000   # 定义数据库中的向量数量 (database size)
        nq = 10     # 定义查询向量的数量 (number of queries)
        np.random.seed(1234) # 设置随机种子以确保结果可复现
        # 创建随机的数据库向量 (float32 类型，Faiss 通常需要这个类型)
        xb = np.random.random((nb, d)).astype("float32")
        # 创建随机的查询向量
        xq = np.random.random((nq, d)).astype("float32")

        # 创建一个标准的 Faiss GPU 资源对象
        # 它管理 GPU 内存和其他 GPU 相关资源
        res = faiss.StandardGpuResources()
        print("已创建 Faiss GPU 资源对象 (faiss.StandardGpuResources)。")

        # 创建一个基础的 CPU Faiss 索引 (这里使用 IndexFlatL2，精确 L2 距离搜索)
        index_flat = faiss.IndexFlatL2(d)
        print(f"已创建 CPU 索引: {type(index_flat)}")

        # 将 CPU 索引复制到 GPU 上
        # faiss.index_cpu_to_gpu(资源对象, GPU设备ID, CPU索引对象)
        # 这里使用设备 ID 0 (第一个 GPU)
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
        print(f"已将索引复制到 GPU 0: {type(gpu_index_flat)}")

        # 检查 GPU 索引的一些属性
        # IndexFlatL2 不需要训练 (is_trained 总是 True)
        print(f"GPU 索引是否需要训练 (is_trained): {gpu_index_flat.is_trained}")

        # 将数据库向量添加到 GPU 索引中
        print(f"正在向 GPU 索引添加 {nb} 个向量...")
        gpu_index_flat.add(xb)
        print(f"向量添加完成。索引中的向量总数: {gpu_index_flat.ntotal}")

        # 在 GPU 索引上执行相似性搜索
        k = 4 # 设置要查找的最近邻居数量
        print(f"正在使用 {nq} 个查询向量在 GPU 索引上搜索最近的 {k} 个邻居...")
        # search 方法返回两个数组：
        # D: 距离数组 (shape: nq x k)，包含每个查询向量到其 k 个最近邻居的距离
        # I: 索引数组 (shape: nq x k)，包含每个查询向量的 k 个最近邻居在原始数据库中的索引
        D, I = gpu_index_flat.search(xq, k)
        print("GPU 搜索完成。")
        # 可以取消注释下面两行来查看具体的搜索结果
        # print("最近邻居的索引 (I):\n", I)
        # print("对应的距离 (D):\n", D)

        # 如果所有步骤都成功执行，则认为 Faiss GPU 功能基本可用
        print("Faiss GPU 功能检查通过。")

    elif ngpu == 0:
        # 如果 Faiss 未检测到 GPU
        print("Faiss 未检测到任何 GPU。无法执行 GPU 功能测试。")
    else: # ngpu > 0 但是 is_available is False
        # 如果 Faiss 检测到 GPU，但之前的 PyTorch 检查失败了
        print("Faiss 检测到了 GPU，但之前的 PyTorch CUDA 检查失败。可能存在环境配置问题。")

# 捕获在 Faiss GPU 检查过程中可能发生的任何异常
except Exception as e:
    print(f"Faiss GPU 功能检查失败，错误: {e}")
    # 导入 traceback 模块并打印详细的错误堆栈信息，帮助诊断问题
    import traceback
    traceback.print_exc()