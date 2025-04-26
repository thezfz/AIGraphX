# tests/vectorization/test_embedder.py
# -*- coding: utf-8 -*-
"""
文件目的：测试 `aigraphx.vectorization.embedder.TextEmbedder` 类。

本测试文件 (`test_embedder.py`) 专注于验证 `TextEmbedder` 类的功能，该类负责加载预训练的
句子转换器模型 (SentenceTransformer) 并使用它来将文本（单个文本或批量文本）转换为数值向量（嵌入）。

测试策略主要是 **单元测试**：
- 使用 `pytest` 和 `unittest.mock` 来模拟（Mock）`TextEmbedder` 的核心依赖项：`sentence_transformers.SentenceTransformer` 类。
- 定义各种模拟模型行为的 Fixtures，包括成功加载、加载失败、编码时出错、返回无效维度等场景。
- 测试 `TextEmbedder` 的初始化逻辑：
    - 能否使用显式传入的模型名称和设备进行初始化。
    - 能否从配置（通过模拟 settings）中读取模型名称和设备进行初始化。
    - 能否使用默认配置进行初始化。
    - 在底层模型加载失败时的处理。
- 测试核心功能：
    - `get_embedding_dimension()`: 获取嵌入向量的维度，包括模型加载失败或返回无效维度的情况。
    - `embed(text)`: 嵌入单个文本，包括处理有效文本、无效输入（None, 空字符串）以及底层模型编码出错的情况。
    - `embed_batch(texts)`: 批量嵌入文本列表，包括处理有效批次、包含无效条目的批次、空批次以及底层模型编码出错的情况。
- 使用 `@pytest.fixture` 创建可重用的测试设置（如模拟模型实例、`TextEmbedder` 实例）。
- 使用 `unittest.mock.patch` 来在测试期间替换掉真实的 `SentenceTransformer` 类或 `settings` 对象。

这些测试确保 `TextEmbedder` 类能够正确加载模型、处理各种输入、生成符合预期的向量，并能优雅地处理初始化和编码过程中的错误。
"""

import pytest  # 导入 pytest 测试框架
import numpy as np  # 导入 numpy 用于处理数值数组（嵌入向量）
import os  # 导入 os 模块（虽然在此文件中可能未直接使用）
from unittest.mock import (
    patch,
    MagicMock,
    call,
    AsyncMock,
)  # 从 unittest.mock 导入模拟工具
from typing import (
    Generator,
    Union,
    Optional,
    List,
    Any,
    Dict,
    Type,
    Callable,
)  # 导入类型提示
import torch  # 导入 torch (SentenceTransformer 可能依赖它)
from typing import cast  # 导入 cast 用于类型转换（主要用于告知类型检查器）
import importlib  # 导入 importlib，可能用于重新加载模块（例如配置）

# 假设测试是从项目根目录运行的
from aigraphx.vectorization.embedder import TextEmbedder  # 导入被测试的 TextEmbedder 类
from aigraphx.core import config  # 导入配置模块，可能用于获取默认设置或重新加载


# --- Fixtures ---
# Pytest fixtures 用于创建可重用的测试设置和资源。


@pytest.fixture(
    scope="module"
)  # scope="module" 表示此 fixture 在整个测试模块中只执行一次
def MockSentenceTransformer() -> Type:
    """
    Pytest Fixture: 提供一个模拟的 SentenceTransformer 类本身。

    这个 fixture 返回的是一个 *类*，而不是类的实例。
    这主要用于在测试中断言 `embedder_instance.model` 的类型，
    或者在 `patch` 时提供一个可调用的对象。

    Returns:
        Type: 一个模拟 SentenceTransformer 行为的类。
    """

    class MockST:  # 定义一个内部类来模拟 SentenceTransformer
        def __init__(
            self,
            model_name: str,  # 模拟构造函数接收模型名称
            device: Optional[str] = None,  # 模拟接收设备参数
            trust_remote_code: bool = False,  # 模拟接收信任远程代码参数
        ) -> None:
            """模拟 SentenceTransformer 的构造函数。"""
            self.model_name = model_name  # 存储模型名称
            self.device = device or "cpu"  # 存储设备，默认为 "cpu"
            self._dimension = 384  # 设置一个默认的模拟嵌入维度
            # print(f"模拟 SentenceTransformer 使用模型 '{model_name}' 在设备 '{self.device}' 上初始化") # 调试打印

        def get_sentence_embedding_dimension(self) -> int:
            """模拟获取嵌入维度的方法。"""
            return self._dimension

        def encode(
            self,
            sentences: Union[str, List[str]],  # 输入可以是单个字符串或字符串列表
            convert_to_numpy: bool = True,  # 模拟其他参数
            normalize_embeddings: bool = False,
            show_progress_bar: bool = False,
        ) -> np.ndarray:
            """模拟编码（嵌入）方法。"""
            # print(f"模拟 encode 被调用: {sentences}") # 调试打印
            if isinstance(sentences, str):  # 如果输入是单个字符串
                # 返回一个形状为 (dimension,) 的全 1 numpy 数组
                return np.ones(self._dimension, dtype=np.float32)
            elif isinstance(sentences, list):  # 如果输入是列表
                # 返回一个形状为 (len(sentences), dimension) 的全 1 numpy 数组
                return np.ones((len(sentences), self._dimension), dtype=np.float32)
            else:
                raise TypeError("模拟 encode 期望输入是 str 或 list")

    return MockST  # 返回这个模拟类


@pytest.fixture
def mock_st_model() -> Generator[MagicMock, None, None]:
    """Pytest Fixture: 提供一个*能正常工作*的 SentenceTransformer 模拟*实例*。"""
    mock_model = MagicMock()  # 创建一个通用的模拟对象
    # 配置 encode 方法的返回值（一个简单的 numpy 数组）
    mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    # 配置 get_sentence_embedding_dimension 方法的返回值
    mock_model.get_sentence_embedding_dimension.return_value = 3
    yield mock_model  # 使用 yield 返回模拟对象，测试结束后会自动清理


@pytest.fixture
def mock_st_model_encode_error() -> Generator[MagicMock, None, None]:
    """Pytest Fixture: 提供一个在调用 encode 方法时会*抛出错误*的 SentenceTransformer 模拟实例。"""
    mock_model = MagicMock()
    # 配置 encode 方法的 side_effect 为抛出一个 RuntimeError
    mock_model.encode.side_effect = RuntimeError("Embedding failed")
    # 仍然需要配置维度方法
    mock_model.get_sentence_embedding_dimension.return_value = 3
    yield mock_model


@pytest.fixture
def mock_st_model_bad_dimension() -> Generator[MagicMock, None, None]:
    """Pytest Fixture: 提供一个 get_sentence_embedding_dimension 方法返回*无效值 (None)* 的 SentenceTransformer 模拟实例。"""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    # 配置维度方法返回 None，模拟模型未能提供维度信息的情况
    mock_model.get_sentence_embedding_dimension.return_value = None
    yield mock_model


@pytest.fixture
def embedder_instance(
    MockSentenceTransformer: Type,  # 请求模拟的 SentenceTransformer 类 fixture
) -> Generator[TextEmbedder, None, None]:
    """
    Pytest Fixture: 提供一个使用*能正常工作*的模拟模型初始化的 TextEmbedder 实例。
    这个 embedder 使用显式传入的模型名称。
    """
    # 使用 patch 上下文管理器，在执行期间将真实的 SentenceTransformer 类替换为我们的模拟类
    with patch(
        "aigraphx.vectorization.embedder.SentenceTransformer", MockSentenceTransformer
    ) as MockSTClass:  # MockSTClass 是被 patch 后的模拟类
        # 在 patch 的作用域内实例化 TextEmbedder
        # 它会调用被 patch 后的 SentenceTransformer (即 MockSentenceTransformer)
        instance = TextEmbedder(model_name="mock-model", device="cpu")  # 显式传入参数
        yield instance  # 返回创建的实例，测试结束后 patch 会自动撤销


@pytest.fixture
def embedder_instance_default(
    MockSentenceTransformer: Type,  # 请求模拟类
) -> Generator[TextEmbedder, None, None]:
    """
    Pytest Fixture: 提供一个使用*默认*模型名称初始化的 TextEmbedder 实例。
    它仍然使用模拟的 SentenceTransformer 类。
    """
    with patch(
        "aigraphx.vectorization.embedder.SentenceTransformer", MockSentenceTransformer
    ) as MockSTClass:
        # 初始化 TextEmbedder 时不传入 model_name，让它尝试从配置或默认值加载
        # 传入 device="cpu" 确保测试环境一致性
        # 即使 TextEmbedder 尝试读取配置，底层的 SentenceTransformer 仍然是我们的模拟类
        instance = TextEmbedder(device="cpu")
        yield instance


# --- 以下 fixtures (mock_model, mock_settings, embedder) 是另一种模拟方式， ---
# --- 可能与上面的 embedder_instance* fixtures 有重叠，选择一种方式即可 ---


@pytest.fixture
def mock_model() -> MagicMock:
    """Pytest Fixture: 提供一个配置好的 SentenceTransformer 模拟模型实例。"""
    model = MagicMock()  # 创建模拟对象

    # 定义一个更复杂的 mock encode 方法，可以处理单个字符串和列表
    def mock_encode(
        sentences: Union[str, List[str]],
        batch_size: int = 32,  # 模拟接收 batch_size 参数
        show_progress_bar: bool = False,  # 模拟接收进度条参数
        **kwargs: Any,  # 接收其他可能的关键字参数
    ) -> np.ndarray:
        if isinstance(sentences, str):
            return np.array([0.1, 0.2, 0.3])  # 单个文本返回 1D 数组
        elif isinstance(sentences, list):
            # 列表返回 2D 数组，行数等于列表长度
            return np.array([[0.1, 0.2, 0.3]] * len(sentences))
        else:
            raise TypeError("输入必须是 str 或 list of str")

    # 将这个函数设置为 encode 方法的 side_effect
    model.encode.side_effect = mock_encode
    # 模拟维度方法
    model.get_sentence_embedding_dimension.return_value = 3
    # 如果模型可能被直接调用 (如 model(text))，可以模拟 __call__ 方法
    # model.__call__ = model.encode
    return model


@pytest.fixture
def mock_settings() -> MagicMock:
    """Pytest Fixture: 提供一个模拟的 settings 对象。"""
    settings = MagicMock()
    # 设置模拟的配置项值
    settings.EMBEDDING_MODEL_NAME = "mock-model"  # 模拟模型名称
    settings.EMBEDDING_BATCH_SIZE = 2  # 使用较小的批次大小进行测试
    settings.EMBEDDING_DEVICE = "cpu"  # 模拟设备
    return settings


@pytest.fixture
def embedder(
    mock_model: MagicMock,  # 请求上面定义的模拟模型实例 fixture
    mock_settings: MagicMock,  # 请求上面定义的模拟 settings fixture
) -> Generator[TextEmbedder, None, None]:
    """
    Pytest Fixture: 使用模拟的模型和设置来创建 TextEmbedder 实例。
    这是另一种创建测试用 embedder 实例的方式。
    """
    # 使用 patch 同时替换 SentenceTransformer 类和 settings 对象
    with (
        patch(
            # 目标是 embedder 模块中的 SentenceTransformer
            "aigraphx.vectorization.embedder.SentenceTransformer",
            return_value=mock_model,  # 配置 patch 后的类返回我们准备好的 mock_model 实例
        ) as _,  # 不需要使用 patch 后的模拟类本身
        patch(
            # 目标是 embedder 模块中的 settings 对象
            "aigraphx.vectorization.embedder.settings",
            mock_settings,  # 替换为我们准备好的 mock_settings 实例
        ) as _,
    ):
        # 在 patch 的作用域内实例化 TextEmbedder
        # 它会使用被 patch 的 settings 和 SentenceTransformer
        embedder = TextEmbedder(
            model_name=mock_settings.EMBEDDING_MODEL_NAME,  # 从模拟设置获取名称
            device=mock_settings.EMBEDDING_DEVICE,  # 从模拟设置获取设备
        )
        # 以前的代码可能直接在实例上设置属性，但现在 TextEmbedder 内部处理
        # yield embedder 返回实例供测试使用
        yield embedder


# --- 测试用例 ---


def test_embedder_init_explicit(
    embedder_instance: TextEmbedder,  # 请求使用模拟类初始化的实例
    MockSentenceTransformer: Type,  # 请求模拟类本身用于类型检查
) -> None:
    """测试：使用显式传入的参数初始化 TextEmbedder。"""
    assert embedder_instance.model_name == "mock-model", "模型名称不匹配"
    assert embedder_instance.device == "cpu", "设备名称不匹配"
    assert embedder_instance.model is not None, "模型实例不应为 None"
    # 验证加载的模型确实是我们的模拟类的实例
    assert isinstance(embedder_instance.model, MockSentenceTransformer), (
        "加载的模型类型不正确"
    )


def test_embedder_init_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试：TextEmbedder 能否从（模拟的）配置/环境变量中读取模型名称进行初始化。"""
    # --- 准备 ---
    expected_model_name = "mock-model-from-env"  # 预期的模型名称

    # 创建一个临时的模拟 settings 对象
    mock_settings = MagicMock()
    mock_settings.sentence_transformer_model = expected_model_name  # 设置模型名称
    mock_settings.embedder_device = "cpu"  # 设置设备

    # 创建一个模拟的 SentenceTransformer 类（返回一个模拟实例）
    mock_st_class = MagicMock()
    mock_model = MagicMock()
    mock_st_class.return_value = mock_model

    # --- 执行与 Patch ---
    # 使用 patch 同时替换 embedder 模块中的 settings 和 SentenceTransformer
    with (
        patch(
            "aigraphx.vectorization.embedder.settings", mock_settings
        ),  # 替换 settings
        patch(
            "aigraphx.vectorization.embedder.SentenceTransformer", mock_st_class
        ),  # 替换类
    ):
        # 在 patch 生效期间实例化 TextEmbedder，不传入 model_name
        embedder = TextEmbedder()

        # --- 断言 ---
        # 验证实例属性是否从模拟 settings 中获取
        assert embedder.model_name == expected_model_name
        assert embedder.device == "cpu"
        assert embedder.model is not None  # 验证模型已加载（是模拟实例）

        # 验证模拟的 SentenceTransformer 类是否以正确的参数被调用
        mock_st_class.assert_called_once_with(
            expected_model_name,  # 应该使用模拟 settings 中的模型名称
            device="cpu",  # 设备
            trust_remote_code=True,  # 验证默认参数
        )


def test_embedder_init_default(
    embedder_instance_default: TextEmbedder,  # 请求使用默认配置初始化的实例
    MockSentenceTransformer: Type,  # 请求模拟类用于类型检查
) -> None:
    """测试：使用默认配置初始化 TextEmbedder。"""
    # 获取实际的默认模型名称（可能来自 .env 文件或 config.py 的默认值）
    expected_default_model = config.settings.sentence_transformer_model
    # 验证实例属性
    assert embedder_instance_default.model_name == expected_default_model, (
        "默认模型名称不匹配"
    )
    assert embedder_instance_default.device == "cpu", "默认设备不匹配"
    assert embedder_instance_default.model is not None, "默认初始化时模型不应为 None"
    # 验证模型类型
    assert isinstance(embedder_instance_default.model, MockSentenceTransformer), (
        "默认加载的模型类型不正确"
    )


# 测试初始化失败的情况
@patch("aigraphx.vectorization.embedder.SentenceTransformer")  # Patch 掉真实的类
def test_text_embedder_init_load_error(MockSentenceTransformer: MagicMock) -> None:
    """测试场景：当 SentenceTransformer 类在初始化时抛出异常。
    预期行为：TextEmbedder 应捕获异常并将 self.model 设置为 None。
    """
    # --- 准备 ---
    # 配置 patch 后的模拟类，使其在被调用时抛出异常
    MockSentenceTransformer.side_effect = Exception("Model loading failed")
    model_name = "bad-model"  # 使用一个假设的坏模型名称

    # --- 执行 ---
    # 实例化 TextEmbedder，预期它会调用模拟类并捕获异常
    embedder = TextEmbedder(model_name=model_name, device="cpu")

    # --- 断言 ---
    # 验证 TextEmbedder 内部的 model 属性是否为 None
    assert embedder.model is None, "模型加载失败时，内部模型应为 None"
    # 验证模拟类确实被以正确的参数调用了一次
    MockSentenceTransformer.assert_called_once_with(
        model_name, device="cpu", trust_remote_code=True
    )


def test_get_embedding_dimension(embedder_instance: TextEmbedder) -> None:
    """测试：成功获取嵌入维度。"""
    # embedder_instance 使用的 MockSentenceTransformer 默认维度是 384
    assert embedder_instance.get_embedding_dimension() == 384, "获取的维度不正确"


# 重命名以反映测试的是初始化失败的场景
@patch("aigraphx.vectorization.embedder.SentenceTransformer")
def test_get_embedding_dimension_model_init_failed(
    MockSentenceTransformer: MagicMock,
) -> None:
    """测试场景：当模型在初始化时就加载失败时，获取嵌入维度。
    预期行为：应返回 0。
    """
    # --- 准备 ---
    # 模拟初始化失败
    MockSentenceTransformer.side_effect = Exception("Load failed")
    # 实例化 Embedder，预期 model 为 None
    embedder = TextEmbedder(model_name="bad-model")
    assert embedder.model is None, "首先确认模型确实为 None"

    # --- 执行与断言 ---
    # 调用获取维度的方法
    dimension = embedder.get_embedding_dimension()
    # 验证返回值为 0
    assert dimension == 0, "模型为 None 时，维度应返回 0"


@patch("aigraphx.vectorization.embedder.SentenceTransformer")  # Patch 掉真实类
def test_get_embedding_dimension_model_returns_none(
    MockSentenceTransformer: MagicMock,  # 注入模拟类
    mock_st_model_bad_dimension: MagicMock,  # 注入维度返回 None 的模拟实例 fixture
) -> None:
    """测试场景：当模型实例的 get_sentence_embedding_dimension 方法返回 None 时。
    预期行为：TextEmbedder 的 get_embedding_dimension 应返回 0。
    """
    # --- 准备 ---
    # 配置模拟类返回那个维度无效的模拟实例
    MockSentenceTransformer.return_value = mock_st_model_bad_dimension
    # 实例化 Embedder
    embedder = TextEmbedder(model_name="test-model")

    # --- 执行 ---
    dimension = embedder.get_embedding_dimension()

    # --- 断言 ---
    # 验证返回值为 0
    assert dimension == 0, "当模型维度为 None 时，应返回 0"
    # 验证底层模型的维度方法确实被调用了
    mock_st_model_bad_dimension.get_sentence_embedding_dimension.assert_called_once()


def test_embed_single_text(embedder_instance: TextEmbedder) -> None:
    """测试：成功嵌入单个有效文本。"""
    # --- 准备 ---
    text = "This is a test."
    # 定义预期的嵌入向量（与 MockSentenceTransformer 的模拟行为一致）
    expected_embedding = np.ones(384, dtype=np.float32)
    # 确保 embedder_instance 的模型不是 None (它应该已经被 fixture 初始化了)
    assert embedder_instance.model is not None

    # --- 执行与 Patch ---
    # 使用 patch.object 精确地 mock *这个实例* 的 model 对象的 encode 方法
    # 这避免了全局 patch，更精确
    with patch.object(
        embedder_instance.model, "encode", return_value=expected_embedding
    ) as mock_encode:  # mock_encode 是 model.encode 的模拟方法
        # 调用 embedder 的 embed 方法
        embedding = embedder_instance.embed(text)

        # --- 断言 ---
        assert embedding is not None, "嵌入结果不应为 None"
        assert isinstance(embedding, np.ndarray), "嵌入结果应为 numpy 数组"
        assert embedding.shape == (384,), "嵌入向量形状应为 (384,)"
        assert embedding.dtype == np.float32, "嵌入向量数据类型应为 float32"
        # 断言我们 patch 的模拟 encode 方法被以正确的参数调用了一次
        mock_encode.assert_called_once_with(
            text,  # 原始文本
            convert_to_numpy=True,  # 默认参数
            normalize_embeddings=True,  # 默认参数 (TextEmbedder 内部设置)
        )


def test_embed_invalid_text(embedder_instance: TextEmbedder) -> None:
    """测试：尝试嵌入无效输入（None 或空字符串）。
    预期行为：应返回 None，并且不调用底层模型的 encode 方法。
    """
    # 使用 patch 来 mock 底层模型的 encode 方法，以便检查它是否被调用
    with patch.object(
        embedder_instance.model,
        "encode",
        return_value=None,  # 返回值不重要，主要是检查调用
    ) as mock_encode:
        # 测试输入为 None
        assert embedder_instance.embed(None) is None, "输入为 None 时应返回 None"
        # 断言 encode 未被调用
        mock_encode.assert_not_called()

        # 测试输入为空字符串
        assert embedder_instance.embed("") is None, "输入为空字符串时应返回 None"
        # 断言 encode 仍然未被调用
        mock_encode.assert_not_called()


# 重命名以反映测试的是初始化失败
@patch("aigraphx.vectorization.embedder.SentenceTransformer")
def test_embed_single_model_init_failed(MockSentenceTransformer: MagicMock) -> None:
    """测试场景：当模型在初始化时加载失败时，调用 embed 方法。
    预期行为：应返回 None。
    """
    # --- 准备 ---
    # 模拟初始化失败
    MockSentenceTransformer.side_effect = Exception("Load failed")
    # 实例化 Embedder，预期 model 为 None
    embedder = TextEmbedder(model_name="bad-model")
    assert embedder.model is None, "首先确认模型为 None"

    # --- 执行与断言 ---
    text = "hello world"
    embedding = embedder.embed(text)
    assert embedding is None, "模型为 None 时，embed 应返回 None"


@patch("aigraphx.vectorization.embedder.SentenceTransformer")  # Patch 真实类
def test_embed_encode_error(
    MockSentenceTransformer: MagicMock,  # 注入模拟类
    mock_st_model_encode_error: MagicMock,  # 注入 encode 会出错的模拟实例
) -> None:
    """测试场景：当底层模型的 encode 方法抛出异常时，调用 embed 方法。
    预期行为：应捕获异常并返回 None。
    """
    # --- 准备 ---
    # 配置模拟类返回那个 encode 会出错的实例
    MockSentenceTransformer.return_value = mock_st_model_encode_error
    # 实例化 Embedder
    embedder = TextEmbedder(model_name="test-model")
    text = "hello world"
    assert embedder.model is not None  # 确认模型已加载

    # --- 执行 ---
    embedding = embedder.embed(text)

    # --- 断言 ---
    # 验证返回值为 None
    assert embedding is None, "当 encode 出错时，embed 应返回 None"
    # 验证底层模型的 encode 方法仍然被调用了（即使它抛出了错误）
    mock_st_model_encode_error.encode.assert_called_once_with(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def test_embed_batch(embedder_instance: TextEmbedder) -> None:
    """测试：成功批量嵌入多个有效文本。"""
    # --- 准备 ---
    texts = ["Text 1", "Text 2", "Text 3"]  # 输入的文本列表
    # 预期的返回结果（根据模拟模型的行为）
    expected_embeddings = np.ones((3, 384), dtype=np.float32)
    assert embedder_instance.model is not None  # 确保模型存在

    # --- 执行与 Patch ---
    # Patch 实例的 model 的 encode 方法
    with patch.object(
        embedder_instance.model, "encode", return_value=expected_embeddings
    ) as mock_encode:
        # 调用 embed_batch
        # 使用 cast 告诉类型检查器 texts 是 List[Optional[str]]，即使我们知道它只包含 str
        embeddings = embedder_instance.embed_batch(cast(List[Optional[str]], texts))

        # --- 断言 ---
        assert embeddings is not None, "批量嵌入结果不应为 None"
        assert isinstance(embeddings, np.ndarray), "结果应为 numpy 数组"
        assert embeddings.shape == (3, 384), "结果形状应为 (3, 384)"
        # 比较数组内容是否相等
        np.testing.assert_array_equal(embeddings, expected_embeddings)
        # 验证底层 encode 方法被调用
        mock_encode.assert_called_once_with(
            texts,  # 传入原始列表
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,  # 假设批次大小未达到显示进度条的阈值
        )


def test_embed_batch_with_invalid(embedder_instance: TextEmbedder) -> None:
    """测试：批量嵌入包含无效条目（None, 空字符串）的列表。
    预期行为：应过滤掉无效条目，只对有效条目调用 encode，并返回有效条目的嵌入结果。
    """
    # --- 准备 ---
    texts: List[Optional[str]] = [  # 包含无效条目的列表
        "Valid 1",
        None,
        "Valid 2",
        "",
        "  ",  # 包含空格的字符串（通常被视为空或无效，取决于具体实现）
    ]
    # 预期实际传递给底层 encode 方法的有效文本列表
    # (假设 TextEmbedder 会过滤掉 None 和空字符串，但保留空格字符串)
    valid_texts_expected_by_encode = [
        "Valid 1",
        "Valid 2",
        "  ",
    ]
    # 预期的嵌入结果（只包含 3 个有效文本的嵌入）
    expected_embeddings = np.ones((3, 384), dtype=np.float32)
    assert embedder_instance.model is not None

    # --- 执行与 Patch ---
    with patch.object(
        embedder_instance.model, "encode", return_value=expected_embeddings
    ) as mock_encode:
        embeddings = embedder_instance.embed_batch(texts)

        # --- 断言 ---
        assert embeddings is not None, "结果不应为 None"
        # 结果数组的形状应只反映有效文本的数量
        assert embeddings.shape == (3, 384), "结果形状应为 (3, 384)"
        np.testing.assert_array_equal(embeddings, expected_embeddings)
        # 验证底层 encode 方法是否只用过滤后的有效文本列表调用
        mock_encode.assert_called_once_with(
            valid_texts_expected_by_encode,  # 验证传入的是过滤后的列表
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


def test_embed_batch_empty(embedder_instance: TextEmbedder) -> None:
    """测试：批量嵌入空列表，或只包含无效条目的列表。
    预期行为：应返回一个形状为 (0, dimension) 的空 numpy 数组，不调用 encode。
    """
    assert embedder_instance.model is not None  # 确保模型存在

    # Patch encode 方法以检查调用情况
    with patch.object(embedder_instance.model, "encode") as mock_encode:
        # 获取嵌入维度
        embedder_dim = embedder_instance.get_embedding_dimension()  # 预期为 384
        assert embedder_dim == 384, "测试前获取的维度不正确"

        # 测试空列表输入
        embeddings_empty = embedder_instance.embed_batch([])
        assert embeddings_empty is not None, "空列表输入时不应返回 None"
        assert isinstance(embeddings_empty, np.ndarray), "空列表输入时应返回 ndarray"
        # 验证返回数组的形状是 (0, dimension)
        assert embeddings_empty.shape == (0, embedder_dim), (
            f"空列表结果形状应为 (0, {embedder_dim})"
        )
        # 验证 encode 未被调用
        mock_encode.assert_not_called()

        # 测试只包含无效条目的列表
        embeddings_all_invalid = embedder_instance.embed_batch([None, "", None])
        assert embeddings_all_invalid is not None, "全无效列表输入时不应返回 None"
        assert isinstance(embeddings_all_invalid, np.ndarray), (
            "全无效列表输入时应返回 ndarray"
        )
        # 验证返回数组的形状也是 (0, dimension)
        assert embeddings_all_invalid.shape == (0, embedder_dim), (
            f"全无效列表结果形状应为 (0, {embedder_dim})"
        )
        # 验证 encode 仍然未被调用
        mock_encode.assert_not_called()


# 重命名以反映测试的是初始化失败
@patch("aigraphx.vectorization.embedder.SentenceTransformer")
def test_embed_batch_model_init_failed(MockSentenceTransformer: MagicMock) -> None:
    """测试场景：当模型在初始化时加载失败时，调用 embed_batch 方法。
    预期行为：应返回 None。
    """
    # --- 准备 ---
    # 模拟初始化失败
    MockSentenceTransformer.side_effect = Exception("Load failed")
    # 实例化 Embedder，预期 model 为 None
    embedder = TextEmbedder(model_name="bad-model")
    assert embedder.model is None, "首先确认模型为 None"

    # --- 执行与断言 ---
    assert embedder.embed_batch(["test1", "test2"]) is None, (
        "模型为 None 时，embed_batch 应返回 None"
    )


@patch("aigraphx.vectorization.embedder.SentenceTransformer")  # Patch 真实类
def test_embed_batch_encode_error(
    MockSentenceTransformer: MagicMock,  # 注入模拟类
    mock_st_model_encode_error: MagicMock,  # 注入 encode 会出错的模拟实例
) -> None:
    """测试场景：当底层模型的 encode 方法抛出异常时，调用 embed_batch 方法。
    预期行为：应捕获异常并返回 None。
    """
    # --- 准备 ---
    # 配置模拟类返回那个 encode 会出错的实例
    MockSentenceTransformer.return_value = mock_st_model_encode_error
    # 实例化 Embedder
    embedder = TextEmbedder(model_name="test-model")
    texts = ["hello", "world"]  # 准备输入列表
    assert embedder.model is not None  # 确认模型已加载

    # --- 执行 ---
    # 调用 embed_batch，使用 cast 辅助类型检查
    embeddings = embedder.embed_batch(cast(List[Optional[str]], texts))

    # --- 断言 ---
    # 验证返回值为 None
    assert embeddings is None, "当 encode 出错时，embed_batch 应返回 None"
    # 验证底层模型的 encode 方法仍然被调用了（即使它抛出了错误）
    mock_st_model_encode_error.encode.assert_called_once_with(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,  # 假设未达到进度条阈值
    )


# --- 以下是之前版本中可能存在的、但与当前 TextEmbedder 接口不符的测试 ---
# --- 保留注释作为参考，说明它们为什么被注释掉 ---

# 以下测试与 TextEmbedder 实际接口不匹配，需要修改或删除
# (例如，TextEmbedder 可能没有名为 `encode` 或 `batch_size` 的公共属性或方法)
# 这里我们假设实际的 TextEmbedder 类没有这些属性或方法，因此注释掉。


def test_embedder_initialization(
    embedder: TextEmbedder, mock_model: MagicMock, mock_settings: MagicMock
) -> None:
    """
    (已弃用/需修改) 测试 TextEmbedder 初始化。
    原意是检查内部属性，但更好的做法是测试类的行为而非内部状态。
    """
    # 验证 SentenceTransformer 是否用正确的模型名称和设备调用
    assert embedder.model_name == mock_settings.EMBEDDING_MODEL_NAME
    assert embedder.model is mock_model  # 验证内部模型实例
    assert embedder.device == mock_settings.EMBEDDING_DEVICE
    # 验证内部属性是否存在（如果 TextEmbedder 有这些属性）
    # assert hasattr(embedder, "_batch_size")
    # assert hasattr(embedder, "_dimension")
    pass  # 替换为 pass 或删除，因为行为已在其他测试中验证


"""
# 以下测试依赖于类中不存在的公共方法或属性，暂时注释掉

def test_encode_single_text(embedder: TextEmbedder, mock_model: MagicMock) -> None:
    # TextEmbedder 没有公共的 encode 方法，应测试 embed 方法。
    pass

def test_encode_list_of_texts(embedder: TextEmbedder, mock_model: MagicMock) -> None:
    # TextEmbedder 没有公共的 encode 方法，应测试 embed_batch 方法。
    pass

def test_embed_batch_single_list(embedder: TextEmbedder, mock_model: MagicMock) -> None:
    # 这个测试用例似乎与 test_embed_batch 重复或不清晰。
    pass

def test_embed_batch_multiple_batches(embedder: TextEmbedder, mock_model: MagicMock) -> None:
    # TextEmbedder 的 embed_batch 内部处理批处理，外部调用者不关心内部如何分批。
    # 这个测试逻辑可能需要重新设计，或者验证 embed_batch 对长列表的处理。
    pass

def test_embed_batch_empty_list(embedder: TextEmbedder, mock_model: MagicMock) -> None:
    # 这个场景已由 test_embed_batch_empty 覆盖。
    pass

def test_batch_embedding_different_types(embedder: TextEmbedder, text: Union[str, List[str]]) -> np.ndarray:
    # 这个测试似乎意图测试 embed 和 embed_batch 的统一性或不同输入类型，
    # 但测试结构不完整，且其逻辑已分别在 test_embed_* 和 test_embed_batch_* 中覆盖。
    pass

# 使用参数化来测试无效输入，但 TextEmbedder 没有公共 encode 方法
@pytest.mark.parametrize(
    "invalid_input",
    [
        123, # Integer
        None, # NoneType
        [1, 2, 3], # List of integers
        ["Valid string", None, "Another string"], # List with None
    ]
)
def test_encode_invalid_input(embedder: TextEmbedder, mock_model: MagicMock, invalid_input: Any) -> None:
    # 应改为测试 embed 或 embed_batch 对无效输入的处理，这已在其他测试中完成。
    pass

# 使用参数化测试 embed_batch 的无效输入
@pytest.mark.parametrize(
    "invalid_input",
    [
        [1, 2, 3], # List of integers - Pydantic/类型提示会先拦截
        # ["Valid string", None, "Another string"], # 这个场景已在 test_embed_batch_with_invalid 中测试
        # [[], "String"], # List with empty list - 输入类型可能不符合预期
    ]
)
def test_embed_batch_invalid_input(embedder: TextEmbedder, mock_model: MagicMock, invalid_input: Any) -> None:
    # 这个测试的部分场景已经被覆盖，部分场景可能因类型提示而无法直接测试，
    # 或者需要更精细地模拟内部过滤逻辑。
    pass

def test_model_loading_failure(mock_settings: MagicMock) -> None:
    # 这个场景已由 test_text_embedder_init_load_error 覆盖。
    pass

def test_get_dimension(embedder: TextEmbedder, mock_model: MagicMock) -> None:
    # 这个场景已由 test_get_embedding_dimension 覆盖。
    pass
"""

# --- 测试用例结束 ---
