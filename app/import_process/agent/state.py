"""导入 LangGraph State 定义。"""

from typing import TypedDict
import copy
from app.core.logger import logger


class ImportGraphState(TypedDict):
    """定义导入工作流中流转的数据结构。

    Attributes:
        task_id: 导入任务唯一标识。
        tenant_id: 租户标识。
        kb_id: 知识库标识。
        user_id: 上传用户标识。
        doc_id: 当前导入文档标识。
        doc_version: 当前导入的文档版本号。
        previous_doc_version: 导入前的 active 文档版本号。
        file_hash: 上传文件内容指纹。
        content_hash: 解析后 Markdown 内容指纹。
        vector_version: 当前导入使用的向量版本。
        chunk_strategy: 当前导入使用的切分策略名称。
        chunk_strategy_version: 当前导入使用的切分策略版本。
        import_action: 当前导入动作，例如 create、update 或 noop。
        skip_reason: 跳过导入时的原因。
        changed_chunks: 新增或变更的 chunks，需要重新生成 embedding。
        reused_chunks: 内容未变化的 chunks，会复用上一 active 版本向量。
        removed_chunks: 上一 active 版本存在但当前版本已删除的 chunks。
        milvus_insert_count: 当前版本写入 Milvus 的 chunk 数量。
    """
    task_id: str  # 任务唯一ID，用于追踪日志
    tenant_id: str  # 租户标识
    kb_id: str  # 知识库标识
    user_id: str  # 上传用户标识
    doc_id: str  # 文档标识
    doc_version: int  # 当前导入版本
    previous_doc_version: int | None  # 上一个active版本
    file_hash: str  # 上传文件指纹
    content_hash: str  # 解析后内容指纹
    vector_version: str  # 向量版本
    chunk_strategy: str  # 切分策略名称
    chunk_strategy_version: str  # 切分策略版本
    import_action: str  # create/update/noop等导入动作
    skip_reason: str  # noop或失败时的原因
    changed_chunks: list  # P1-B预留：新增或变更切片
    reused_chunks: list  # P1-B预留：复用切片
    removed_chunks: list  # P1-B预留：删除切片
    milvus_insert_count: int  # Milvus实际写入数量

    # --- 流程控制标记 ---
    is_md_read_enabled: bool  # 是否启用 Markdown 读取路径
    is_pdf_read_enabled: bool  # 是否启用 PDF 读取路径

    # --- 切块相关 ---
    is_normal_split_enabled: bool
    is_silicon_flow_api_enabled: bool
    is_advanced_split_enabled: bool
    is_vllm_enabled: bool

    # --- 路径相关 ---
    local_dir: str  # 当前工作目录或输出目录
    local_file_path: str  # 原始输入文件路径
    file_title: str  # 文件标题（文件名去后缀）
    pdf_path: str  # PDF 文件路径 (如果输入是PDF)
    md_path: str  # Markdown 文件路径 (转换后或直接输入的)
    split_path: str  # 分块后的文件路径
    embeddings_path: str  # 向量数据库文件路径

    # --- 内容数据 ---
    md_content: str  # Markdown 的全文内容
    chunks: list  # 切片后的文本列表，包含 metadata
    item_name: str  # 识别出的主体名称 (如: "万用表")，用于增强检索

    # --- 数据库相关 ---
    embeddings_content: list  # 包含向量数据的列表，准备写入 Milvus


# 建议定一个初始化对象，方便后续使用
# 定义图状态的默认初始值
graph_default_state: ImportGraphState = {
    "task_id": "",
    "tenant_id": "default",
    "kb_id": "default",
    "user_id": "anonymous",
    "doc_id": "",
    "doc_version": 0,
    "previous_doc_version": None,
    "file_hash": "",
    "content_hash": "",
    "vector_version": "bge_m3_v1_legacy",
    "chunk_strategy": "markdown_heading",
    "chunk_strategy_version": "v1",
    "import_action": "create",
    "skip_reason": "",
    "changed_chunks": [],
    "reused_chunks": [],
    "removed_chunks": [],
    "milvus_insert_count": 0,
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "is_normal_split_enabled": True,
    "is_silicon_flow_api_enabled": True,
    "is_advanced_split_enabled": False,
    "is_vllm_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "split_path": "",
    "embeddings_path": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "embeddings_content": []
}


def create_default_state(**overrides) -> ImportGraphState:
    """创建默认状态，支持覆盖。

    Args:
        **overrides: 要覆盖的字段（关键字参数解包）。

    Returns:
        新的状态实例。

    Examples:
        state = create_default_state(task_id="task_001", local_file_path="doc.pdf")
    """

    # 默认状态
    state = copy.deepcopy(graph_default_state)
    # 用 overrides 覆盖默认值
    state.update(overrides)
    # 返回创建好的状态字典实例
    return state


def get_default_state() -> ImportGraphState:
    """返回一个新的状态实例。

    Returns:
        导入流程默认状态副本。
    """
    return copy.deepcopy(graph_default_state)


if __name__ == "__main__":
    """
    测试
    """
    # 创建默认状态
    state = create_default_state(local_file_path="万用表RS-12的使用.pdf")
    logger.info(state)
