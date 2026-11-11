"""Markdown 文档切分节点。

该模块负责将导入流程中的 Markdown 全文切分为适合向量化和检索的 chunks。
"""

import json
import os
import re
import sys


from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task
# from opentelemetry import trace

# tracer = trace.get_tracer(__name__)
# _provider = trace.get_tracer_provider()
# print(f"🔥 [DEBUG-TRACER] provider 类型: {type(_provider).__name__}")
# print(f"🔥 [DEBUG-TRACER] provider: {_provider}")

# --- 配置参数 (Configuration) ---
# 单个Chunk最大字符长度：超过则触发二次切分（适配大模型上下文窗口）
MAX_CONTENT_LENGTH = 2000  # 512 - 1500 token
# 短Chunk合并阈值：同父标题的短Chunk会被合并，减少碎片化
MIN_CONTENT_LENGTH = 500  # 最小的长度


def step_1_get_content(state: ImportGraphState):
    """
    获取并标准化 Markdown 内容。

    Args:
        state: 导入流程的 LangGraph 状态。

    Returns:
        标准化换行后的 Markdown 内容和文件标题。

    Raises:
        Exception: Markdown 内容为空时抛出。
    """
    md_content = state["md_content"]
    file_title = state.get("file_title", "default_title")
    if not md_content:
        logger.error(f"[step_1_get_content]没有有效的md内容，直接抛出异常！！！！")
        raise Exception("请检查输入文件路径是否正确！！")

    new_md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    return new_md_content, file_title


def step_2_split_content(md_content, md_file_name):
    """
    根据 Markdown 标题对文档做语义粗切分。

    Args:
        md_content: Markdown 文件内容。
        md_file_name: Markdown 文件名或文档标题。

    Returns:
        sections 和识别到的标题数量，sections 元素包含 content、title、file_title、parent_title。
    """
    # 创建子span - 会自动关联到当前父span
    # with tracer.start_as_current_span("step.split_by_headings") as span:
    #     span.set_attribute("md.file.name", md_file_name)
    #     span.set_attribute("content.length", len(md_content))

    # 1.对md_content 先按行切分
    md_content_lines = md_content.split("\n")
    # 2.对每一行进行判断，如果以 ## 开头，则表示是一个标题,还需判断不是代码块
    # 2.1 准备正则表达式
    # 正则说明：
    # \s 表示空白字符 \s* 表示匹配任意空白字符，包括换行符 \s+ 表示匹配至少一个空白字符 \s* 表示匹配任意空白字符，包括换行符
    # #{1,6}表示md文档标题的级别，1-6级
    # \s+ 表示在#号后匹配至少1个空格，和标题内容区分开
    # .+ 表示匹配任意字符，包括换行符,作为标题内容
    title_pattern = re.compile(r'^\s*(#{1,6})\s+(.+?)\s*$')

    # 2.2 定义临时变量
    current_title = ""
    current_parent_title = md_file_name
    current_lines = []  # 当前标题下的内容行
    title_stack = []
    title_count = 0
    is_code_block = False
    # 2.3 最终存储的列表  sections = []
    sections = []

    def append_current_section():
        # 标题切换或文档结束时再落盘，避免同一章节在逐行读取时被反复追加。
        if not current_lines or not any(line.strip() for line in current_lines):
            return
        section_title = current_title or md_file_name
        sections.append({
            "content": "\n".join(current_lines).strip(),
            "title": section_title,
            "file_title": md_file_name,
            "parent_title": current_parent_title or section_title
        })

    for line in md_content_lines:
        new_line = line.rstrip()
        stripped_line = new_line.strip()

        # 判断是否是代码块
        if stripped_line.startswith("```") or stripped_line.startswith("~~~"):
            # 进入代码块 或者 退出代码块
            # 第一次来一定进入代码块
            is_code_block = not is_code_block
            # 进入代码块
            current_lines.append(new_line)
            continue

        title_match = title_pattern.match(new_line)
        if not is_code_block and title_match:
            # 先检查（是不是第一次）只要不是第一次，就应该先存储
            append_current_section()

            title_level = len(title_match.group(1))
            title_text = title_match.group(2).strip()
            # 维护 Markdown 标题层级路径，给检索结果保留更完整的语义上下文。
            title_stack[:] = [(level, text) for level, text in title_stack if level < title_level]
            title_stack.append((title_level, title_text))
            current_title = new_line
            current_parent_title = " > ".join(title for _, title in title_stack)
            current_lines = [new_line]
            title_count += 1
        else:
            current_lines.append(new_line)

    append_current_section()

    # 3. 返回结果 sections
    logger.info(f"已经完成chunks的语义粗切！识别标题数量：{title_count}, section数量:{len(sections)}")
    # return sections, title_count, len(md_content_lines)
    # 添加事件记录关键结果
    # span.add_event("split.complete", {
    #     "section.count": title_count,
    #     "line.count": len(md_content_lines)
    # })
    return sections, title_count


def split_long_content(section, splitter):
    """
    将超长 section 二次切分为多个子块。

    Args:
        section: 粗切分后的 section。
        splitter: 递归字符切分器。

    Returns:
        二次切分后的子 section 列表。
    """
    content = section["content"]
    chunks = splitter.split_text(content)
    sub_sections = []
    for index, chunk in enumerate(chunks, 1):
        sub_sections.append({
            "content": chunk,
            "file_title": section["file_title"],
            "parent_title": section.get("parent_title", section["title"]),
            "title": f"{section['title']}_{index}",
            "part": index
        })
    return sub_sections


def merge_short_sections(sections, min_chunk_size, max_chunk_size):
    """
    合并相邻且属于同一父标题的短 section。

    Args:
        sections: 待合并的 section 列表。
        min_chunk_size: 触发短块合并的最小字符阈值。
        max_chunk_size: 合并后允许的最大字符阈值。

    Returns:
        合并短块后的 section 列表。
    """
    final_sections = []
    pre_section = None
    for section in sections:
        if not pre_section:
            pre_section = section
            continue
        content = section.get("content", "")
        pre_content = pre_section.get("content", "")
        merged_length = len(pre_content) + len(content) + 1
        need_merge_current = len(pre_content) < min_chunk_size or len(content) < min_chunk_size
        pre_parent_title = pre_section.get("parent_title", pre_section.get("title"))
        current_parent_title = section.get("parent_title", section.get("title"))
        same_parent_title = pre_parent_title and pre_parent_title == current_parent_title
        can_merge_size = merged_length <= max_chunk_size
        if need_merge_current and same_parent_title and can_merge_size:
            # 短块只和同一父标题下的相邻块合并，避免跨章节混入无关语义。
            pre_section["content"] += "\n" + content
            pre_section['part'] = section.get("part")  # 1 <- 2
        else:
            # 添加
            final_sections.append(pre_section)
            pre_section = section
    # 遍历完成后，最后一次添加到结果中
    if pre_section:
        final_sections.append(pre_section)
    return final_sections


def step_3_refine_chunks(sections, max_chunk_size, min_chunk_size):
    """
    对粗切分 section 做精细切分和短块合并。

    Args:
        sections: 粗切分后的 section 列表。
        max_chunk_size: 单个 chunk 的最大字符数。
        min_chunk_size: 短 chunk 合并阈值。

    Returns:
        精细切分后的 section 列表。
    """

    # with tracer.start_as_current_span("step.refine_chunks") as span:
    #     span.set_attribute("section.count", len(sections))
    #     span.set_attribute("max.chunk.size", max_chunk_size)

        # 记录初始块分布
        # initial_sizes = [len(s["content"]) for s in sections]
        # span.set_attribute("initial.sizes.min", min(initial_sizes))
        # span.set_attribute("initial.sizes.max", max(initial_sizes))

    final_sections = []  # 存储处理后的块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=100,
        separators=['\n\n', '\n', '。', '！', '？', "；", ";", "，", ",", " ", ""]
    )

    # 1.遍历所有的块，大的细切
    for section in sections:
        content = section["content"]
        if len(content) < max_chunk_size:
            logger.info(
                f"[split_long_section]:{section.get('title', '')}当前chunk长度{len(content)}"
                f"小于等于{max_chunk_size}，不做二次切割！"
            )
            final_sections.append(section)
        else:
            final_sections.extend(split_long_content(section, splitter))

    # 2. 遍历所有的块，合并小的块
    final_sections = merge_short_sections(final_sections, min_chunk_size, max_chunk_size)

    # 补全属性和参数 part parent_title -> 向量数据库 -》 报错 （split_long_section）
    for section in final_sections:
        section['part'] = section.get('part', 1)
        section['parent_title'] = section.get('parent_title', section.get('title'))

    # 返回即可
    return final_sections


def step_4_backup_and_modify_chunks(state: ImportGraphState, final_sections):
    """
    将切分后的 chunks 备份到本地 JSON 文件。

    Args:
        state: 导入流程的 LangGraph 状态。
        final_sections: 切分完成的 chunks 列表。

    Returns:
        None。
    """
    local_dir = state.get("local_dir") + "/" + state["file_title"]
    backup_file_path = os.path.join(local_dir, "chunks.json")
    with open(backup_file_path, "w", encoding="utf-8") as f:
        json.dump(
            final_sections,  # 将什么数据写到指定的文件流！
            f,  # 写出的位置
            ensure_ascii=False,  # 中文直接原文存储
            indent=4  # json带有缩进 4
        )
    logger.info(f"已经将内容,进行备份到:{backup_file_path}")


def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    将 Markdown 文档切分为适合检索的 chunks。

    Args:
        state: 导入流程的 LangGraph 状态，需包含 md_content、file_title、local_dir 等字段。

    Returns:
        写入 chunks 后的导入流程状态。

    Raises:
        Exception: 文档内容无效或备份 chunks 失败时抛出。
    """

    # 1.节点入口处打印日志以及前端埋点
    func_name = sys._getframe().f_code.co_name
    logger.info(f"【{func_name}】节点开始执行")
    add_running_task(state["task_id"], func_name)

    try:

        # 1.参数校验以及获取内容
        md_content, md_file_name = step_1_get_content(state)

        # 2. 粗粒度切割（md）语义完善===>使用标题切割
        #  return: [{content,title,file_title}]
        sections, title_count = step_2_split_content(md_content, md_file_name)

        # 3.特殊场景，一个文档没有标题，给个默认标题
        if title_count == 0:
            sections = [{
                "content": md_content,
                "title": md_file_name,
                "file_title": md_file_name,
                "parent_title": md_file_name
            }]

        # 4. 细粒度切割（md），大小和重叠合适： 大--->(设置重叠)小 ||  小--->合并
        final_sections = step_3_refine_chunks(sections, MAX_CONTENT_LENGTH, MIN_CONTENT_LENGTH)

        # 此处的chunks 是大小合适，语义完整的，接下来将chunks保存
        state["chunks"] = final_sections
        # 5.数据的备份和chunks属性的修改
        step_4_backup_and_modify_chunks(state, final_sections)
    except Exception as e:
        logger.error(f"【{func_name}】节点执行异常，异常信息为:{e}")
        raise e
    finally:
        # 节点出口处打印日志以及前端埋点
        logger.info(f"【{func_name}】节点执行完毕")
        add_done_task(state["task_id"], func_name)

    return state


if __name__ == '__main__':
    """
       单元测试：联合node_md_img（图片处理节点）进行集成测试
       测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
       测试流程：先运行图片处理→再运行文档切分，验证端到端流程
       """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output/hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "hak180产品安全手册",
            "local_dir": os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")
