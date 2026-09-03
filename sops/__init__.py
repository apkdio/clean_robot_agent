"""SOP（标准操作流程）模块。

对外暴露会话状态与执行入口，并在此注册各业务场景 SOP。
"""

from sops.base import (  # noqa: F401
    SOPS,
    register,
    start_sop,
    continue_sop,
    end_sop,
    has_active_sop,
    get_active_sop_id,
    match_sop,
    handle_followup,
)

# 导入各 SOP 模块，触发 register 注册
import sops.purchase  # noqa: F401
import sops.repair  # noqa: F401
import sops.service_point  # noqa: F401
