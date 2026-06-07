from .alfworld import build_alfworld_obligations
from .webshop import build_webshop_obligations


def build_obligations(env_name: str, traj_uid: str, task_text: str, info: dict | None = None):
    env_l = env_name.lower()
    if "webshop" in env_l:
        return build_webshop_obligations(traj_uid=traj_uid, task_text=task_text, info=info or {})
    return build_alfworld_obligations(traj_uid=traj_uid, task_text=task_text, info=info or {})
