# -*- coding: utf-8 -*-
"""
Embedder - 文本向量化模块

这个文件定义了 `TextEmbedder` 类，它的核心职责是处理文本到向量（嵌入）的转换。
你可以把它想象成一个"翻译器"，把人类理解的文本"翻译"成计算机可以进行数学运算和比较的数字向量。

主要功能:
1.  **加载模型**: 从预先训练好的模型库（比如 Hugging Face Hub）下载并加载一个指定的"文本理解"模型。这个模型通常被称为 Sentence Transformer，因为它擅长理解句子的含义。我们使用的具体模型名称是在配置文件中指定的（比如 `all-MiniLM-L6-v2` 或中文的 `bge-large-zh-v1.5`）。
2.  **错误处理**: 如果模型下载失败、文件损坏或配置不正确，这个类需要能够捕捉到这些错误并记录下来，防止整个程序崩溃。
3.  **单个文本编码**: 提供一个方法，输入一段文本（比如一个搜索查询），输出一个代表这段文本语义的 NumPy 数组（向量）。
4.  **批量文本编码**: 提供一个更高效的方法，一次性处理很多段文本（比如一批论文摘要），将它们都转换成向量，并返回一个包含所有向量的 NumPy 矩阵。
5.  **获取维度**: 提供一个方法，告诉你这个模型产生的向量有多少个数字（即向量的维度）。
6.  **配置读取**: 它需要知道使用哪个模型以及在哪个硬件上运行（比如普通的 CPU，或者更快的 GPU 如 CUDA，或苹果的 MPS）。这些信息从项目的中央配置文件 (`aigraphx.core.config.settings`) 读取。
7.  **设备支持**: 确保模型可以在指定的硬件上运行，以获得最佳性能。

这个文件如何与其他部分协作:
- **读取配置 (`aigraphx.core.config.settings`)**: 从这里获取要加载的模型名称 (`SENTENCE_TRANSFORMER_MODEL`) 和运行模型的设备 (`EMBEDDER_DEVICE`)。
- **被搜索服务使用 (`aigraphx.services.search_service.SearchService`)**: `SearchService` 是处理搜索请求的核心逻辑。当需要进行"语义搜索"（根据意思搜索，而不是简单的关键词匹配）时，它会请求 `TextEmbedder` 把用户的搜索查询和数据库中的文档都转换成向量，然后比较这些向量的相似度。
- **被索引构建脚本使用 (`scripts/sync_pg_to_faiss.py`, `scripts/sync_pg_to_models_faiss.py`)**: 在项目启动前，需要预先处理所有论文摘要和模型描述，将它们转换成向量并存储在一个叫做 Faiss 的高效索引库中。这些脚本会使用 `TextEmbedder` 的批量处理功能来完成这个任务。
- **被依赖注入系统使用 (`aigraphx.api.v1.dependencies`)**: FastAPI 的依赖注入系统在创建 `SearchService` 实例时，会自动将一个已经初始化好的 `TextEmbedder` 实例"注入"给 `SearchService`，这样 `SearchService` 就可以直接使用它了，无需关心 `TextEmbedder` 是如何创建的。
- **可能被其他需要文本向量化的地方使用**: 例如，未来如果需要计算模型描述之间的相似度，也可能会用到这个类。
"""

# 导入 logging 模块，用于记录程序运行过程中的信息、警告和错误。
# 这对于调试和监控程序状态非常重要。
import logging

# 导入 typing 模块中的 List, Optional, Sequence 类型提示。
# 类型提示可以帮助开发者理解函数期望接收什么类型的参数以及返回什么类型的值，
# 也能让静态类型检查工具 (如 MyPy) 发现潜在的类型错误。
# List: 表示列表。
# Optional[X]: 表示这个变量可以是类型 X，也可以是 None。
# Sequence[X]: 表示一个序列（如 list, tuple），其元素是类型 X。
from typing import List, Optional, Sequence

# 导入 numpy 库，并使用别名 np。
# NumPy 是 Python 进行科学计算的核心库，特别擅长处理多维数组（包括向量和矩阵）。
# 我们用它来存储和操作文本嵌入向量。
import numpy as np

# 从 sentence_transformers 库导入 SentenceTransformer 类。
# 这个库专门用于处理句子/文本嵌入，提供了方便的接口来加载和使用预训练模型。
from sentence_transformers import SentenceTransformer

# 导入 os 模块，提供与操作系统交互的功能。
# 虽然这里注释掉了直接加载 .env 文件的代码 (因为现在由配置模块统一处理)，
# 但在其他地方可能仍需 os 模块来处理文件路径等。
import os
# # 不再需要手动加载 .env 文件，配置由 core.config 统一处理
# # from dotenv import load_dotenv

# 导入项目核心配置模块中的 settings 对象。
# 这个 settings 对象包含了从环境变量或 .env 文件加载的所有配置项，
# 例如要使用的模型名称、数据库连接信息等。
# 这样做的好处是配置集中管理，易于修改和维护。
from aigraphx.core.config import settings

# # 移除旧的 dotenv 加载逻辑
# # dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
# # load_dotenv(dotenv_path=dotenv_path)

# 获取一个日志记录器实例，通常使用当前模块的名称 (__name__) 来命名。
# 这样可以方便地追踪日志信息的来源。
logger = logging.getLogger(__name__)


# 定义 TextEmbedder 类
class TextEmbedder:
    """
    这个类封装了与文本嵌入模型相关的所有操作。
    它的主要任务是加载一个预训练的 Sentence Transformer 模型，
    并提供方法将文本转换成数值向量（嵌入）。

    可以把它看作是文本向量化的"引擎"。
    """

    # 类的初始化方法，当创建 TextEmbedder 类的实例时被调用。
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        初始化 Embedder 实例。

        在初始化过程中，它会确定要加载的模型名称和运行设备，
        然后调用内部的 `_load_model` 方法来实际加载模型。

        Args:
            model_name (Optional[str]): 指定要加载的 Sentence Transformer 模型的名称。
                                        例如 'all-MiniLM-L6-v2' 或 'bge-large-zh-v1.5'。
                                        如果创建实例时不提供这个参数（或者提供 None），
                                        它会从项目的全局配置 `settings.sentence_transformer_model` 中读取。
            device (Optional[str]): 指定模型应该在哪个硬件设备上运行。
                                    常见选项有 'cpu'（中央处理器）, 'cuda'（NVIDIA GPU）, 'mps'（Apple Silicon GPU）。
                                    如果创建实例时不提供这个参数（或者提供 None），
                                    它会首先尝试从全局配置 `settings.embedder_device` 读取，
                                    如果配置中也没有，则设为 None，此时 sentence-transformers 库会尝试自动选择最合适的设备。
        """
        # 确定模型名称：优先使用传入的 model_name 参数，如果未传入，则使用全局配置 settings.sentence_transformer_model 的值。
        # `or` 操作符在这里用于提供默认值。
        self.model_name: str = model_name or settings.sentence_transformer_model
        # 确定运行设备：优先使用传入的 device 参数，如果未传入，则使用全局配置 settings.embedder_device 的值。
        self.device: Optional[str] = device or settings.embedder_device

        # 将模型属性初始化为 None。实际的模型对象将在 _load_model 方法中被加载并赋值给这个属性。
        self.model: Optional[SentenceTransformer] = None
        # 在初始化结束前，调用私有方法 _load_model() 来尝试加载模型。
        self._load_model()

    # 定义一个私有方法 _load_model，用于加载模型。
    # 方法名前的下划线 `_` 是一种约定，表示这个方法主要供类内部使用。
    def _load_model(self) -> None:
        """
        这是一个内部方法，负责实际加载 Sentence Transformer 模型。
        它会检查模型是否已经被加载，如果没有，则进行加载。
        这样可以避免在程序运行期间重复加载同一个模型，节省时间和资源。
        """
        # 检查 self.model 是否仍然是 None。如果是，说明模型还没有被加载。
        if self.model is None:
            # 记录一条日志信息，说明即将开始加载模型，以及模型的名称和目标设备。
            # `self.device or 'auto'` 表示如果 device 是 None，就显示 'auto'，表示自动选择。
            logger.info(
                f"开始加载 Sentence Transformer 模型: {self.model_name} 到设备: {self.device or 'auto'}"
            )
            try:
                # 这是加载模型的关键步骤。
                # 调用 SentenceTransformer 类的构造函数，传入模型名称和设备。
                # `device=self.device`: 指定运行设备。如果为 None，库会自动选择。
                # `trust_remote_code=True`: 这是一个安全相关的参数。有些模型（特别是较新的或社区贡献的）可能需要在 Hugging Face Hub 上执行它们仓库中包含的一些自定义 Python 代码才能正确加载。设置 True 表示我们信任这些代码。对于官方或广泛使用的模型，通常不需要设置或可以设为 False。但为了兼容性，这里设置为 True。
                self.model = SentenceTransformer(
                    self.model_name, device=self.device, trust_remote_code=True
                )
                # 如果代码执行到这里，说明模型加载成功，记录一条成功的日志信息。
                logger.info(f"模型 {self.model_name} 加载成功。")
                # （可选）可以取消注释下面两行，在加载成功后立即获取并记录模型的嵌入维度。
                # dim = self.get_embedding_dimension()
                # logger.info(f"模型嵌入维度: {dim}")
            # 使用 except 块来捕获在加载模型过程中可能发生的任何异常。
            # 例如，网络连接失败导致模型下载不了、模型文件损坏、或者 `trust_remote_code=True` 时执行的远程代码出错等。
            except Exception as e:
                # 如果发生异常，记录一条包含异常信息的错误日志。
                # `logger.exception` 会自动记录完整的错误堆栈信息，非常有助于调试。
                logger.exception(
                    f"加载 Sentence Transformer 模型 '{self.model_name}' 失败: {e}"
                )
                # 在这里可以选择是否将异常重新抛出。
                # 如果重新抛出 (`raise e`)，那么 TextEmbedder 初始化失败会中断程序的启动，这在某些情况下是期望的行为（例如，如果嵌入功能是程序的核心，没有它无法运行）。
                # 如果不重新抛出（像现在这样注释掉），程序会继续运行，但 self.model 会保持为 None，后续尝试使用模型的操作会失败或返回 None。
                # raise e

    # 定义一个公共方法 get_embedding_dimension，用于获取模型产生的嵌入向量的维度。
    def get_embedding_dimension(self) -> int:
        """
        返回当前加载的 Sentence Transformer 模型的嵌入向量维度（即向量包含多少个数字）。

        例如，`all-MiniLM-L6-v2` 模型的维度是 384。

        Returns:
            int: 模型的嵌入维度。如果模型没有成功加载，或者获取维度时出错，返回 0。
                 返回 0 是一个明确的信号，表示无法获取有效的维度。
        """
        # 首先检查 self.model 是否已经被成功加载。
        if self.model is None:
            # 如果模型是 None，记录错误日志并返回 0。
            logger.error("嵌入模型未加载，无法获取维度。")
            return 0
        try:
            # 调用 sentence_transformers 库提供的 `get_sentence_embedding_dimension` 方法获取维度。
            dimension = self.model.get_sentence_embedding_dimension()
            # 检查获取到的维度是否有效。在某些罕见情况下，它可能返回 None。
            if dimension is None:
                logger.error("模型返回的嵌入维度为 None。")
                return 0
            # 如果维度有效，返回该整数值。
            return dimension
        # 捕获在调用 `get_sentence_embedding_dimension` 时可能发生的任何异常。
        except Exception as e:
            # 如果发生异常，记录错误日志并返回 0。
            logger.exception(f"获取嵌入维度时出错: {e}")
            return 0

    # 定义一个公共方法 embed，用于将单个文本字符串转换为嵌入向量。
    def embed(self, text: Optional[str]) -> Optional[np.ndarray]:
        """
        将单个文本字符串编码（嵌入）成一个 NumPy 向量。

        Args:
            text (Optional[str]): 需要嵌入的文本。它可以是一个字符串，也可以是 None。

        Returns:
            Optional[np.ndarray]: 一个 NumPy 数组，表示文本的嵌入向量。
                                  如果发生以下情况，则返回 None:
                                  - 输入的 `text` 是 None、空字符串或不是字符串类型。
                                  - 嵌入模型没有成功加载 (`self.model` is None)。
                                  - 在嵌入过程中发生错误。
        """
        # 检查模型是否已加载。
        if self.model is None:
            logger.warning("嵌入模型未加载。无法执行嵌入操作。")
            return None
        # 检查输入是否是一个有效的、非空的字符串。
        # `not text` 会捕获 None 和空字符串 ""。
        # `not isinstance(text, str)` 确保输入是字符串类型。
        if not text or not isinstance(text, str):
            # 如果输入无效，记录一条调试信息（通常只在调试模式下显示），然后返回 None。
            logger.debug(f"无效的嵌入输入文本: {text}")
            return None

        try:
            # 调用模型的 `encode` 方法来执行实际的编码工作。
            embedding = self.model.encode(
                text,  # 要编码的文本
                convert_to_numpy=True,  # 指示方法返回 NumPy 数组，而不是 PyTorch 或 TensorFlow 张量。
                normalize_embeddings=True,  # 对生成的嵌入向量进行 L2 归一化 (使向量长度为 1)。
                # 这对于后续使用余弦相似度比较向量非常重要，可以确保相似度度量不受向量长度的影响。
            )
            # 返回计算得到的嵌入向量。
            return embedding
        # 捕获在调用 `encode` 时可能发生的任何异常。
        except Exception as e:
            # 如果发生异常，记录错误日志。
            # 为了避免日志过长，这里只记录了输入文本的前 50 个字符。
            logger.exception(f"嵌入文本 '{text[:50]}...' 时出错: {e}")
            # 返回 None 表示嵌入失败。
            return None

    # 定义一个公共方法 embed_batch，用于将一批（多个）文本字符串转换为嵌入向量。
    # 处理批量数据通常比逐个处理效率更高。
    def embed_batch(self, texts: Sequence[Optional[str]]) -> Optional[np.ndarray]:
        """
        将一个文本字符串序列（例如列表）高效地编码（嵌入）成一个 NumPy 矩阵。
        矩阵的每一行对应输入序列中一个文本的嵌入向量。

        Args:
            texts (Sequence[Optional[str]]): 包含要嵌入的文本字符串的序列（可以是列表、元组等）。
                                            这个序列可以包含 None 或者无效的条目，方法内部会处理。

        Returns:
            Optional[np.ndarray]: 一个 NumPy 矩阵 (二维数组)，形状为 (n, d)，
                                  其中 n 是输入序列中有效文本的数量，d 是模型的嵌入维度。
                                  如果发生以下情况，则返回 None:
                                  - 嵌入模型没有成功加载 (`self.model` is None)。
                                  - 在嵌入过程中发生错误。
                                  - 输入序列过滤后没有有效的文本，并且无法获取模型维度来创建空的正确形状的数组。
                                  如果输入序列过滤后没有有效的文本，但模型已加载且能获取维度，
                                  则返回一个形状为 (0, d) 的空 NumPy 数组。
        """
        # 检查模型是否已加载。
        if self.model is None:
            logger.warning("嵌入模型未加载。无法执行批量嵌入操作。")
            return None

        # 对输入的 `texts` 序列进行过滤，只保留有效的、非空的字符串。
        # 使用列表推导式高效地完成这个操作。
        valid_texts = [t for t in texts if t and isinstance(t, str)]

        # 检查过滤后的 `valid_texts` 列表是否为空。
        if not valid_texts:
            # 如果列表为空，说明输入序列要么是空的，要么只包含无效条目。
            logger.debug("输入文本列表为空或不包含有效字符串。")
            # 尝试获取模型维度，以便返回一个形状正确的空数组。
            dim = self.get_embedding_dimension()
            if dim > 0:
                # 如果成功获取维度，创建一个形状为 (0, dim) 的空 NumPy 数组并返回。
                # 这表示没有嵌入任何内容，但结果的形状是符合预期的。
                return np.empty(
                    (0, dim), dtype=np.float32
                )  # 指定 dtype 为 float32，与模型输出一致
            else:
                # 如果获取维度失败（例如模型加载有问题），记录错误并返回 None。
                logger.error("无法确定嵌入维度，无法为无效输入创建空数组。")
                return None

        # 如果 `valid_texts` 不为空，则继续执行批量编码。
        try:
            # 决定是否在编码过程中显示进度条。
            # 如果处理的文本数量较多（这里设定阈值为 50），显示进度条可以提供更好的用户反馈。
            show_progress_bar = len(valid_texts) > 50

            # 调用模型的 `encode` 方法进行批量编码。
            embeddings = self.model.encode(
                valid_texts,  # 包含有效文本的列表
                convert_to_numpy=True,  # 返回 NumPy 矩阵
                normalize_embeddings=True,  # 对所有嵌入向量进行归一化
                show_progress_bar=show_progress_bar,  # 根据前面的判断决定是否显示进度条
                # batch_size=32 # （可选）可以取消注释并调整 batch_size 参数来控制每次传递给模型的文本数量。
                # 调整这个值可能会影响性能和内存使用，需要根据硬件进行实验。默认值通常比较合理。
            )
            # 返回包含所有嵌入向量的 NumPy 矩阵。
            return embeddings
        # 捕获在批量调用 `encode` 时可能发生的任何异常。
        except Exception as e:
            # 如果发生异常，记录错误日志。
            # 只记录第一个有效文本的前 50 个字符作为代表，避免日志过长。
            logger.exception(
                f"批量嵌入以 '{valid_texts[0][:50]}...' 开头的文本时出错: {e}"
            )
            # 返回 None 表示批量嵌入失败。
            return None

    # 定义一个公共方法 embed_query，专门用于嵌入搜索查询。
    def embed_query(self, query: str) -> Optional[np.ndarray]:
        """
        为搜索查询生成嵌入向量。

        在当前的实现中，这个方法直接调用通用的 `embed` 方法。
        但是，单独定义这个方法提供了一个扩展点：
        某些嵌入模型（例如一些为信息检索优化的模型）建议在处理查询时，
        在文本前添加特定的前缀（如 "query: "），而在处理文档时添加另一个前缀（如 "passage: "）。
        如果将来使用这类模型，可以在这个方法里方便地添加查询前缀的逻辑。

        Args:
            query (str): 用户输入的搜索查询字符串。

        Returns:
            Optional[np.ndarray]: 查询文本的嵌入向量 (NumPy 数组)。
                                  如果嵌入失败或模型未加载，返回 None。
        """
        # 示例：如果将来使用的模型需要在查询前加前缀，可以在这里添加逻辑
        # if self.model_name in ["some-model-requiring-prefix"]: # 假设模型名称列表
        #     query = f"query: {query}" # 添加 "query: " 前缀

        # 目前直接调用通用的 embed 方法来处理查询。
        return self.embed(query)

    # 定义一个公共方法 embed_documents，专门用于嵌入文档（如论文摘要、模型描述等）。
    def embed_documents(
        self, documents: Sequence[Optional[str]]
    ) -> Optional[np.ndarray]:
        """
        为一批文档（例如论文摘要、模型描述等）生成嵌入向量矩阵。

        与 `embed_query` 类似，这个方法目前直接调用通用的 `embed_batch` 方法。
        它同样提供了一个扩展点，用于将来可能需要的文档特定处理（例如添加 "passage: " 前缀）。

        Args:
            documents (Sequence[Optional[str]]): 包含要嵌入的文档字符串的序列。

        Returns:
            Optional[np.ndarray]: 包含所有有效文档嵌入向量的 NumPy 矩阵。
                                  如果批量嵌入失败或模型未加载，返回 None。
        """
        # 示例：如果将来使用的模型需要在文档前加前缀
        # processed_docs = [] # 创建一个新列表来存储处理过的文档
        # for doc in documents: # 遍历输入的文档序列
        #     if doc and isinstance(doc, str): # 检查是否是有效字符串
        #         # 如果模型需要前缀
        #         if self.model_name in ["some-model-requiring-prefix"]:
        #             processed_docs.append(f"passage: {doc}") # 添加 "passage: " 前缀
        #         else:
        #             processed_docs.append(doc) # 不需要前缀，直接添加
        #     # 对于无效条目 (None 或非字符串)，可以选择忽略，或者添加一个占位符/错误处理
        #     # else:
        #     #     processed_docs.append(None) # 或者根据需要处理

        # # 然后调用 embed_batch 处理加工后的 processed_docs 列表
        # # return self.embed_batch(processed_docs)

        # 目前直接调用通用的 embed_batch 方法来处理文档。
        return self.embed_batch(documents)


# Example usage (for testing)
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     embedder = TextEmbedder() # Uses default model from env or fallback
#     if embedder.model:
#         print(f"Model Dim: {embedder.get_embedding_dimension()}")
#         text1 = "This is the first test sentence."
#         text2 = "This is a second, different sentence."
#         text3 = "First test sentence is this one."
#         invalid_text = ""

#         emb1 = embedder.embed(text1)
#         emb2 = embedder.embed(text2)
#         emb3 = embedder.embed(text3)
#         emb_invalid = embedder.embed(invalid_text)

#         print("Embedding 1:", emb1[:5], "... Shape:", emb1.shape if emb1 is not None else None)
#         print("Embedding 2:", emb2[:5], "... Shape:", emb2.shape if emb2 is not None else None)
#         print("Embedding 3:", emb3[:5], "... Shape:", emb3.shape if emb3 is not None else None)
#         print("Embedding Invalid:", emb_invalid)

#         # Calculate cosine similarity (requires numpy)
#         if emb1 is not None and emb3 is not None:
#             similarity = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
#             # Since we use normalize_embeddings=True, dot product is enough
#             similarity_dot = np.dot(emb1, emb3)
#             print(f"Similarity (1 vs 3) - Dot Product: {similarity_dot:.4f}")

#         if emb1 is not None and emb2 is not None:
#             similarity_dot_12 = np.dot(emb1, emb2)
#             print(f"Similarity (1 vs 2) - Dot Product: {similarity_dot_12:.4f}")

#         # Batch embedding
#         batch_texts = [text1, text2, text3, "One more sentence.", invalid_text]
#         batch_embeddings = embedder.embed_batch(batch_texts)
#         print("Batch Embeddings Shape:", batch_embeddings.shape if batch_embeddings is not None else None)
