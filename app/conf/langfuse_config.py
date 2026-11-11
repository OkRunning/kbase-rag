"""读取 Langfuse 可选观测配置。"""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LangFuseConfig:
    """Langfuse 客户端配置。

    Attributes:
        langfuse_secret_key: Langfuse secret key。
        langfuse_public_key: Langfuse public key。
        langfuse_base_url: Langfuse 服务地址。
        org_id: Langfuse 组织 ID。
    """

    langfuse_secret_key: Optional[str]
    langfuse_public_key: Optional[str]
    langfuse_base_url: Optional[str]
    org_id: Optional[str]

    # LANGFUSE_SECRET_KEY = "sk-lf-11b548a0-6f8c-45a6-a67e-5c659d70b16b"
    # LANGFUSE_PUBLIC_KEY = "pk-lf-774ce72a-a88d-428e-81ee-384c6825e3db"
    # LANGFUSE_BASE_URL = "http://localhost:3000"


langfuse_config = LangFuseConfig(
    langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    langfuse_base_url=os.getenv("LANGFUSE_BASE_URL"),
    org_id=os.getenv("LANGFUSE_ORG_ID")
)
