import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

from app.conf.mineru_config import mineru_config
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.task_utils import add_running_task, add_done_task


"""
node_pdf_to_md

step_1_valid_path

step_2_get_parsed_file_url

step_3_download_zip_file
"""

def step_1_valid_path(state):
    """
        进行路径校验！ pdf_path失效 直接异常处理!
                    local_dir 没有，给与默认值
        :param state:
    """
    logger.debug("【step_1_valid_path】在pdf转md处理，开始文件校验")
    pdf_path = state.get("pdf_path")
    local_dir = state.get("local_dir")

    if not pdf_path:
        logger.info(f"【step_1_valid_path】节点处理，上传文件为空，无法解析")
        raise ValueError("【step_1_valid_path】节点处理，上传文件为空，无法解析")

    if not local_dir:
        local_dir = str(PROJECT_ROOT / "output")
        logger.info(f"step_1_validate_paths检查发现local_dir没有赋值，给与默认值：{local_dir}！")

    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)
    if not pdf_path_obj.exists():
        logger.info(f"【step_1_valid_path】节点处理，上传文件不存在，无法解析")
        raise FileNotFoundError("【step_1_valid_path】节点处理，上传文件不存在，无法解析")

    if not local_dir_obj.exists():
        local_dir_obj.mkdir(parents=True, exist_ok=True)

    return pdf_path_obj, local_dir_obj


def step_2_get_parsed_file_url(pdf_path_obj) -> str:
    """
    通过MinerU上传pdf，并解析，返回解析后的url
    :param pdf_path_obj: 上传解析pdf文件的path对象
    :return: minerU解析后的md压缩包 地址
    """

    # 1. 申请 上传解析的地址
    # 1.1 申请需要的条件
    token = mineru_config.api_key
    url = mineru_config.base_url + "/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name": f"{pdf_path_obj.name}"}
        ],
        "model_version": "vlm"
    }

    # 1.2 通过post请求获取解析文件地址
    response = requests.post(url, headers=header, json=data)

    # 1.3  获取解析结果
    if response.status_code != 200 or response.json()["code"] != 0:
        logger.error(f"【step_2_get_parsed_file_url】节点处理，获取解析文件地址失败，请检查minerU服务是否正常")
        raise RuntimeError("【step_2_get_parsed_file_url】节点处理，获取解析文件地址失败，请检查minerU服务是否正常")

    upload_file_url = response.json()["data"]["file_urls"][0]
    # 用于后续获取本次请求的结果
    batch_id = response.json()["data"]["batch_id"]

    # 2.上传解析文件

    # 使用put请求将文件pdf_path_obj 上传到指定url
    # 不能直接使用put！ 这块很大可能会报错！
    # 原因： 电脑如果开了各种代理，会给put请求添加一些额外的请求头！ 将文件上传到第三方的文件存储服务器
    #       文件服务器检查都比较严格！ 会拒绝存储！ 导致报错
    # get、post一般都是宽进宽出，put是严进严出
    http_session = requests.Session()
    http_session.trust_env = False  # 1.禁止走代理，2.复用请求对象

    try:

        with open(pdf_path_obj, 'rb') as f:
            file_date = f.read()
        res_upload = http_session.put(upload_file_url, data=file_date)
        if res_upload.status_code != 200:
            logger.error(f"【step_2_get_parsed_file_url】节点处理，上传文件失败，请检查minerU服务是否正常")
            raise RuntimeError("【step_2_get_parsed_file_url】节点处理，上传文件失败，请检查minerU服务是否正常")
    except Exception as e:
        logger.error(f"【step_2_get_parsed_file_url】节点处理，上传文件失败，请检查minerU服务是否正常")
        raise RuntimeError("【step_2_get_parsed_file_url】节点处理，上传文件失败，请检查minerU服务是否正常")
    finally:
        http_session.close()

    # 3. 轮询获取解析结果。确保获取到结果
    url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"

    # 设计一个循环，3秒获取一次，最多获取10分钟。 官方600s可解析600页pdf
    timeout_seconds = 600  # 超时时间 600s
    poll_interval = 3  # 间隔3秒
    start_time = time.time()
    while True:
        # 3.1 超时判断，不能站在第一次的角度！ 站在一个宏观的角度
        if time.time() - start_time > timeout_seconds:
            logger.error(f"【step_2_get_parsed_file_url】节点处理，获取解析结果超时，请检查minerU服务是否正常")
            raise TimeoutError("【step_2_get_parsed_file_url】节点处理，获取解析结果超时，请检查minerU服务是否正常")
        # 3.2 向url获取解析结果
        res = requests.get(url, headers=header)

        # 3.3 解析结果并获取结果url
        if res.status_code != 200:
            if 500 <= res.status_code < 600:
                time.sleep(poll_interval)
                continue
            raise RuntimeError(f"【step_2_get_parsed_file_url】节点处理，获取解析结果失败，返回状态码{res.status_code}")

        json_data = res.json()
        if json_data["code"] != 0:
            logger.error(f"【step_2_get_parsed_file_url】节点处理，获取解析结果失败，失败信息为：{json_data['msg']}")
            raise RuntimeError("【step_2_get_parsed_file_url】节点处理，获取解析结果失败，失败信息为：{json_data['msg']}")

        extract_result = json_data["data"]["extract_result"][0]
        result_state = extract_result["state"]
        if result_state == "done":
            full_url = extract_result['full_zip_url']
            logger.info(
                f"【step_2_get_parsed_file_url】节点处理，获取解析结果成功,耗时{time.time() - start_time}秒，解析结果{full_url}")
            # 获取解析结果
            return full_url
        elif result_state == "failed":
            logger.error(f"【step_2_get_parsed_file_url】节点处理，获取解析结果失败，失败信息为：{extract_result['err_msg']}")
            raise RuntimeError(
                f"【step_2_get_parsed_file_url】节点处理，获取解析结果失败，失败信息为：{extract_result['message']}")
        else:
            time.sleep(poll_interval)


def step_3_download_zip_file(zip_url, local_dir_obj, stem) -> str:
    """
    下载指定md.zip文件，并且解压，返回解压后的md文件地址
    :param zip_url: zip文件的url
    :param local_dir_obj: 本地目录对象
    :param stem: 文件存储子路径
    :return: 下载后的md文件路径
    """
    # 1. 下载zip文件，返回response流
    response = requests.get(zip_url)
    if response.status_code != 200:
        logger.error(f"【step_3_download_zip_file】节点处理，下载zip文件失败，请检查minerU服务是否正常")
        raise RuntimeError("【step_3_download_zip_file】节点处理，下载zip文件失败，请检查minerU服务是否正常")

    # 2. 将文件存储到本地
    zip_save_path = local_dir_obj / f"{stem}_result.zip"
    with open(zip_save_path, 'wb') as f:
        # response.content 获取响应内容
        f.write(response.content)

    logger.info(f"【step_3_download_zip_file】节点处理，下载zip文件成功，保存路径为：{zip_save_path}")
    # 3.清空旧目录
    extract_dir = local_dir_obj / stem
    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    extract_dir.mkdir(parents=True, exist_ok=True)

    # 4.解压： 使用zipfile模块解压
    # zipfile 处理 zip文件的模块
    # zipfile.ZipFile(zip_save_path, 'r') as zip_ref: 创建一个zip对象，只能读取，即解压
    with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
        # 对zip对象进行解压，指定解压目录
        zip_ref.extractall(extract_dir)

    # 5. 返回文件地址
    # 解压后的文件名，可能叫 文件名.md， 也可能叫full.md
    md_file_list = list(extract_dir.rglob("*.md"))

    if not md_file_list:
        logger.error(f"【step_3_download_zip_file】节点处理,没有找到md文件，请检查路径输入是否正确")
        raise RuntimeError("【step_3_download_zip_file】节点处理,没有找到md文件，请检查路径输入是否正确")

    target_md_file = None
    # 检查有没有原文件名.md
    for md_file in md_file_list:
        if md_file.name == f"{stem}.md":
            target_md_file = md_file
            break
        elif md_file.name.islower() == "full.md":
            target_md_file = md_file
            break

    if not target_md_file:
        target_md_file = md_file_list[0]

    if target_md_file.stem != stem:
        # target_md_file.with_name(f"{stem}.md") 修改path对象（不涉及文件操作），返回结果是修改后的path对象
        # target_md_file.rename(...) 修改磁盘中的文件名称，return 新的路径path
        target_md_file = target_md_file.rename(target_md_file.with_name(f"{stem}.md"))

    #返回文件的绝对路径
    final_md_str_path = str(target_md_file.resolve())
    logger.info(f"【step_3_download_zip_file】节点处理，解压zip文件成功，保存路径为：{final_md_str_path}")
    return final_md_str_path


def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。

    1. 进入的日志和任务状态的配置
    2. 进行参数校验
            local_file_path->已完成参数非空，接下来要验证文件是否真实存在
            local_dir->判断文件路径是否存在，不存在给予默认值且创建目录

    3. 调用 MinerU (magic-pdf) 工具，返回一个下载文件的地址 xx.zip url地址
    4. 下载zip包，并且解析和提取 （local_dir）
    5. 把md_path地址进行赋值，读取md的文件内容 md_content赋值（文本内容）
    6. 结束的日志和任务状态的配置

    注意：此节点需要加异常处理
    """

    # 1.节点入口处打印日志以及前端埋点
    func_name = sys._getframe().f_code.co_name
    logger.info(f"【{func_name}】节点开始执行，")
    add_running_task(state["task_id"], func_name)
    try:

        # 2.参数校验
        pdf_path_obj, local_dir_obj = step_1_valid_path(state)

        # 3.调用mineru工具，并返回下载文件地址
        zip_url = step_2_get_parsed_file_url(pdf_path_obj)

        # 4.下载zip包，并解析
        md_path = step_3_download_zip_file(zip_url, local_dir_obj, pdf_path_obj.stem)

        # 5. 把md_path地址进行赋值，读取md的文件内容 md_content赋值（文本内容）
        # 更新数据
        state["md_path"] = str(md_path)
        state['local_dir'] = str(local_dir_obj)  # 主要处理下！是str类型

        with open(md_path, "r", encoding="utf-8") as f:
            state["md_content"] = f.read()

    except Exception as e:
        logger.error(f"【{func_name}】节点调用mineru解析文件处理异常，异常信息为:{e}")
        raise
    finally:
        # 6.节点处理完成后，打印日志以及前端埋点
        logger.info(f"【{func_name}】节点节点执行结束，")
        add_done_task(state["task_id"], func_name)
    return state


if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("testdoc", "个人贷款管理办法.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")
