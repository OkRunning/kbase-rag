import base64
import os
import re
import sys
from pathlib import Path
from typing import Tuple, Deque

from minio.deleteobjects import DeleteObject

from app.clients.minio_utils import get_minio_client
from app.conf.lm_config import lm_config
from app.conf.minio_config import minio_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.lm_utils import get_llm_client
from app.utils.rate_limit_utils import apply_api_rate_limit
from app.utils.task_utils import add_running_task, add_done_task

"""
    主要目的：将md文件中的图片进行单独处理，方便模型识别图片的含义
    主要动作： 
            1.图片-》文件服务器-〉图片网络地址 
            
            2.（上文100）图片（下文100） -》视觉模型--〉图片总结
            将1.2的结果替换md文件中图片保存方式，--》[图片的总结](图片网络地址)
            
            -〉md_content -- 新的内容（图片处理后的） ||md_path 处理后的md地址
            
    总结技术：
        minio
        视觉模式 
    步骤：
        1. 校验本节点数据是否完整
            入参： state  -> md_path md_content 
            返回： 1. 校验后的md_content  2.md路径对象  3. 获取图片的文件夹 images
        
        2. 获取md_content图片上下文，用于下一步的视觉处理
            入参： md_content,images图片路径
            返回： [(图片名,图片地址,(上文,下文))]
        
        3. 调用模型总结图片内容
            入参：第二次的响应 [(图片名,图片地址,(上文,下文))]   || md文件的名称（提示词中 md文件名就是存储图片images的文件名）
            返回： {图片名称：总结}
        
        4. 通过minio上传文件以及更新md_content
            入参：minio_client || {图片名:总结,......} || [(图片名,图片地址,(上文,下文))] (minio) || md_content 旧 || md文件的名称（提示词中 md文件名就是存储图片images的文件名）
            返回：new_md_content
            
        5. 数据的最终处理和备份    
            入参： new_md_content , 原md地址 -》 xx.md -> xx_new.md  
            返回： 新的md的地址 new_md_path  
"""

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def is_supported_image(filename: str) -> bool:
    """
    判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return: 支持返回True，否则False
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def step_1_get_content(state: ImportGraphState) -> Tuple[str, Path, Path]:
    md_content = state.get("md_content")
    md_path = state.get("md_path")
    if not md_path:
        raise ValueError("md_path is required")

    md_path_obj = Path(md_path)

    if not md_content:
        with open(md_path_obj, "r", encoding="utf-8") as f:
            md_content = f.read()
        state['md_content'] = md_content

    images_dir_obj = md_path_obj.parent / "images"
    logger.debug(
        f"【step_1_get_content】节点处理,获取md文件内容处理完成，文件内容{md_content[:100]}，插图路径为：{images_dir_obj}")
    return md_content, md_path_obj, images_dir_obj


def find_image_context(md_content, img, context_length: int = 100):
    """
    从md 内容中找指定图片名称
    :param md_content: md文件内容
    :param img: 图片名称
    :param context_length: 指定查找图片的上下文长度
    :return: (图片上文，图片下文)
    """

    # 定义正则表达式  .*  .*?
    pattern = re.compile(r"!\[.*?\]\(.*?" + img + ".*?\)")
    # results = []  # 存储图片多处使用，上下文不同 ！ 本次暴力处理，获取第一个！
    matched_content = list(pattern.finditer(md_content))
    if not matched_content:
        logger.warning(f"图片：{img} 在md文件中未找到！")
        return None
    match = matched_content[0]
    if match:
        # for match in pattern.finditer(md_content):
        start, end = match.span()  # span获取匹配对象的起始和终止的位置
        # 获取图片的上下文
        previous_context = md_content[max(0, start - context_length): start]
        suffixes_context = md_content[end: min(len(md_content), end + context_length)]
        context = (previous_context, suffixes_context)
        logger.info(f"图片：{img} 上下文为：{context}")
        return context


def step_2_get_img_context(md_content, md_images_dir):
    """
    获取md文件中图片的上下文
    :param md_content: md文件内容
    :param md_images_dir: 图片目录
    :return:
    """
    targets = []
    if not Path.exists(md_images_dir):
        return targets

    for img in os.listdir(md_images_dir):
        # 判断图片格式
        if not is_supported_image(img):
            logger.warning(f"{img} is not supported image file")
            continue
        # 获取图片上下文
        img_context = find_image_context(md_content, img)
        if not img_context:
            logger.warning(f"{img} 上下文在md文件中未找到，上下文为空！")
            continue

        targets.append((img, str(md_images_dir / img), img_context))

    logger.debug(f"【step_2_get_img_context】节点处理,md文件图片信息收集完成，{targets}")
    return targets


def step_3_generate_img_summaries(img_info, stem):
    """
    节点: 图片总结 (node_md_img)
    用于调用视觉大模型，对图片内容进行总结
    返回{图片名称1：总结，图片名称2:总结2...}
    """
    summaries = {}
    # 获取模型
    lm = get_llm_client(model=lm_config.lv_model)

    request_times = Deque()

    # 循环对每张图片处理
    for img_name, img_path, img_context in img_info:
        # 1.限流
        apply_api_rate_limit(request_times=request_times, max_requests=10, window_seconds=20)
        # 2.加载提示词
        prompt = load_prompt("image_summary", root_folder=stem, image_content=img_context)
        with open(img_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            # 直接放图片的网络地址 "url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
                            # base64图片转后的字符串  jpg -> image/jpeg
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        # 调用大模型返回结果
        response = lm.invoke(messages)
        summary = response.content.strip().replace("\n", "")
        summaries[img_name] = summary
        logger.info(f"【step_3_generate_img_summaries】节点处理,图片：{img_name} 总结完成，结果为：{response.content}")

    return summaries


def step_4_upload_img_and_replace_md(summaries, md_content, img_info, stem):
    """
    将图片上传到minio，并替换md文件中的图片内容和url
    :param summaries: 图片总结
    :param md_content: md文件内容
    :param img_info: 图片信息,格式(图片文件名，图片本地存放目录，（图片上文，图片下文）)
    :param stem: md文件名
    :return: 替换后的md_content
    """
    minio_client = get_minio_client()
    # 1. 删除原文件
    # 1.1 获取要删除的对象
    # 注意：一定要去掉prefix最前面的/（如果有的话）
    object_list = minio_client.list_objects(minio_config.bucket_name, prefix=f"{minio_config.minio_img_dir}/{stem}",
                                            recursive=True)
    if object_list:
        # 需要的DeleteObject
        # 1.2 调用方法进行删除即可
        delete_object_list = [DeleteObject(obj.object_name) for obj in object_list]
        errors = minio_client.remove_objects(minio_config.bucket_name, delete_object_list)
        for error in errors:
            logger.warning(f"【step_4_upload_img_and_replace_md】节点处理,删除对象失败，错误信息为：{error}")

        logger.info(f"已经完成{stem}下的对象清空，本次删除了：{len(delete_object_list)}个对象！！！")

    # 2. 上传文件 拿到url列表
    images_url = {}
    for img_name, img_path, _ in img_info:
        try:
            minio_client.fput_object(minio_config.bucket_name,
                                     f"{minio_config.minio_img_dir}/{stem}/{img_name}", img_path, "image/jpeg")
            # 上传完毕以后记录
            # 图片地址 = 协议 + 端点 + 桶名 + 对象名  http://47.94.86.115:9000/ 桶名 / 对象名
            images_url[img_name] = f"http://{minio_config.endpoint}/{minio_config.bucket_name}/{minio_config.minio_img_dir}/{stem}/{img_name}"
            logger.info(
                f"【step_4_upload_img_and_replace_md】节点处理,图片：{img_name} 上传完成，图片url为：{images_url[img_name]}")

            url_test = minio_client.presigned_get_object(minio_config.bucket_name,
                                                         f"{minio_config.minio_img_dir}/{stem}/{img_name}")

        except Exception as e:
            logger.error(f"【step_4_upload_img_and_replace_md】节点处理,上传图片{img_name}失败，错误信息为：{str(e)}")
            continue

    logger.info(f"【step_4_upload_img_and_replace_md】节点处理,图片上传完成")

    # 3. 汇总图片内容
    # summaries = 图片名: 描述
    # images_url= 图片名：url地址
    # 汇总： {图片名:(描述,url地址)}
    image_all_infos = {}
    for img_name, summary in summaries.items():
        if url := images_url.get(img_name):
            image_all_infos[img_name] = (summary, url)

    logger.info(f"【step_4_upload_img_and_replace_md】节点处理,图片汇总完成，结果为：{image_all_infos}")

    # 4. 替换md文件内容
    for img_name, (summary, url) in image_all_infos.items():
        rep = re.compile(r"!\[.*?\]\(.*?" + img_name + ".*?\)")
        md_content = rep.sub(f"![{summary}]({url})", md_content)

    return md_content


def step_5_save_md(new_md_content, md_path_obj):
    """
    完成新的md的磁盘本分，并且返回老地址！
    新的命名  xxx_new.md
    :param new_md_content: 新内容
    :param md_path_obj: 老地址
    :return: 新地址
    """
    new_md_path = os.path.splitext(md_path_obj)[0] + "_new.md"
    with open(new_md_path, "w", encoding="utf-8") as f:
        f.write(new_md_content)

    logger.info(f"【step_5_save_md】节点处理,保存新md文件完成，新md文件地址为：{new_md_path}")
    return new_md_path


def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. (可选) 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。
    """

    # 1.节点入口处打印日志以及前端埋点
    func_name = sys._getframe().f_code.co_name
    logger.info(f"【{func_name}】节点开始执行")
    add_running_task(state["task_id"], func_name)
    try:
        # 获取文件内容
        md_content, md_path_obj, md_images_dir_obj = step_1_get_content(state)

        if not md_images_dir_obj:
            return state

        # 2. 识别md中使用过的图片，采取做下一步（进行图片总结）
        # [(图片名,图片地址,(上文,下文 = 100))]
        img_info = step_2_get_img_context(md_content, md_images_dir_obj)

        # 3. 进行图片内容的总结和处理 （视觉模型）
        # 参数： 第二次的响应 [(图片名,图片地址,(上文,下文))]   || md文件的名称（提示词中 md文件名就是存储图片images的文件名）
        # 响应： {图片名:总结,......}
        summaries = step_3_generate_img_summaries(img_info, md_path_obj.stem)

        # 4.上传图片到MinIO 并且替换md文件中的图片地址以及描述
        # 参数： summaries ||  md_content 文件内容 || img_info 图片文件存放的目录 || stem 文件名
        # 响应： new_md_content
        new_md_content = step_4_upload_img_and_replace_md(summaries, md_content, img_info, md_path_obj.stem)

        # 5. 保存处理后的md文件
        new_md_path = step_5_save_md(new_md_content, md_path_obj)
        state["md_path"] = str(new_md_path)
        state["md_content"] = new_md_content
    except Exception as e:
        logger.error(f"【{func_name}】节点执行异常，错误信息为：{str(e)}")
        raise e
    finally:
        logger.info(f"【{func_name}】节点开始执行")
        add_done_task(state["task_id"], func_name)

    return state


if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT

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
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
