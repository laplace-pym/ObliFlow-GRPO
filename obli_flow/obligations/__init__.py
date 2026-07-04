from .alfworld import build_alfworld_obligations
from .webshop import build_webshop_obligations
from obli_flow.llm_obligations import try_build_llm_obligations
from obli_flow.offline_subtasks import try_build_offline_obligations


def build_obligations(env_name: str, traj_uid: str, task_text: str, info: dict | None = None, config=None):
    offline_obligations = try_build_offline_obligations(env_name=env_name, traj_uid=traj_uid, task_text=task_text, info=info, config=config)
    if offline_obligations is not None:
        return offline_obligations

    llm_obligations = try_build_llm_obligations(env_name=env_name, traj_uid=traj_uid, task_text=task_text, info=info, config=config)
    if llm_obligations is not None:
        return llm_obligations

    env_l = env_name.lower()
    if "webshop" in env_l:
        return build_webshop_obligations(traj_uid=traj_uid, task_text=task_text, info=info or {})
    return build_alfworld_obligations(traj_uid=traj_uid, task_text=task_text, info=info or {})
