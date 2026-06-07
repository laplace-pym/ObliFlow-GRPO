from obli_flow.schema import ObligationNode
from obli_flow.validators import keyword_set


def build_webshop_obligations(traj_uid: str, task_text: str, info: dict | None = None) -> list[ObligationNode]:
    keywords = sorted(keyword_set(task_text))
    return [
        ObligationNode(
            id=f"{traj_uid}:o_goal_parse",
            traj_uid=traj_uid,
            type="goal_parse",
            target={"task": task_text, "keywords": keywords},
            weight=0.5,
            verifier="goal_overlap",
        ),
        ObligationNode(
            id=f"{traj_uid}:o_query_matches_goal",
            traj_uid=traj_uid,
            type="query_matches_goal",
            target={"keywords": keywords},
            weight=0.8,
            verifier="query_overlap",
        ),
        ObligationNode(
            id=f"{traj_uid}:o_inspect_product",
            traj_uid=traj_uid,
            type="inspect_candidate_product",
            target={"keywords": keywords},
            weight=1.0,
            verifier="product_overlap",
        ),
        ObligationNode(
            id=f"{traj_uid}:o_buy_correct_product",
            traj_uid=traj_uid,
            type="buy_correct_product",
            target=True,
            weight=2.0,
            verifier="task_score",
        ),
        ObligationNode(
            id=f"{traj_uid}:o_final_success",
            traj_uid=traj_uid,
            type="final_success",
            target=True,
            weight=3.0,
            verifier="env_success",
        ),
    ]
