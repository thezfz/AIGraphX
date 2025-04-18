import torch
import faiss  # type: ignore[import-untyped]
import numpy as np

print("--- PyTorch GPU Check ---")
is_available = torch.cuda.is_available()
print(f"torch.cuda.is_available(): {is_available}")

if is_available:
    device_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {device_count}")
    for i in range(device_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("PyTorch cannot detect CUDA. Faiss GPU will likely fail.")

print("\n--- Faiss GPU Check ---")
try:
    # Get the number of GPUs Faiss can see
    ngpu = faiss.get_num_gpus()
    print(f"Faiss sees {ngpu} GPUs.")

    if ngpu > 0 and is_available:
        # Create a simple dataset
        d = 64  # dimension
        nb = 1000  # database size
        nq = 10  # nb of queries
        np.random.seed(1234)
        xb = np.random.random((nb, d)).astype("float32")
        xq = np.random.random((nq, d)).astype("float32")

        # Create a standard GPU resource object
        res = faiss.StandardGpuResources()

        # Create a flat L2 index on the first GPU
        index_flat = faiss.IndexFlatL2(d)
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

        print(f"GPU Index Type: {type(gpu_index_flat)}")
        print(f"Index is trained: {gpu_index_flat.is_trained}")

        # Add vectors to the index
        gpu_index_flat.add(xb)
        print(f"Number of vectors in index: {gpu_index_flat.ntotal}")

        # Search the index
        k = 4  # we want 4 nearest neighbors
        D, I = gpu_index_flat.search(xq, k)
        print("Search completed successfully.")
        # print("Nearest neighbors (indices):\n", I)
        # print("Distances:\n", D)
        print("Faiss GPU check PASSED.")
    elif ngpu == 0:
        print("Faiss cannot detect any GPUs.")
    else:
        print("Faiss detects GPUs, but PyTorch check failed earlier.")

except Exception as e:
    print(f"Faiss GPU check FAILED with error: {e}")
    import traceback

    traceback.print_exc()
