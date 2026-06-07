from obli_flow.action_parser import parse_response
from obli_flow.graph_builder import build_flow_graph
from obli_flow.reward import compute_step_credits
from obli_flow.schema import StepRecord


def test_alfworld_success_flow_has_positive_final_credit():
    records = [
        StepRecord(
            uid="u1",
            traj_uid="t1",
            step_id=0,
            env_name="alfworld/AlfredTWEnv",
            task_text="put a cleaned apple in the fridge",
            anchor_obs="You see an apple and a fridge.",
            raw_response="<think>take apple</think><action>pick up apple</action>",
            parsed_action=parse_response("<think>take apple</think><action>pick up apple</action>", "alfworld/AlfredTWEnv"),
            next_obs="You pick up the apple.",
            reward=0.0,
            done=False,
            info={"admissible_commands": ["pick up apple"], "won": False},
            is_action_valid=True,
        ),
        StepRecord(
            uid="u1",
            traj_uid="t1",
            step_id=1,
            env_name="alfworld/AlfredTWEnv",
            task_text="put a cleaned apple in the fridge",
            anchor_obs="You are carrying an apple near a fridge.",
            raw_response="<think>finish</think><action>put apple in fridge</action>",
            parsed_action=parse_response("<think>finish</think><action>put apple in fridge</action>", "alfworld/AlfredTWEnv"),
            next_obs="success",
            reward=10.0,
            done=True,
            info={"admissible_commands": ["put apple in fridge"], "won": True},
            is_action_valid=True,
            episode_reward=10.0,
        ),
    ]
    graph = build_flow_graph(records, "alfworld/AlfredTWEnv")
    result = compute_step_credits(graph, uid="u1")
    assert any(edge.edge_type == "transformation" for edge in graph.edges)
    assert any(edge.edge_type == "consumption" and edge.src.endswith(":next_obs") and edge.dst.endswith(":a1") for edge in graph.edges)
    assert result.credits[-1].delta_phi > 0
    assert result.credits[-1].obligation_coverage > 0


def test_invalid_noop_gets_negative_flow_reward():
    record = StepRecord(
        uid="u1",
        traj_uid="t2",
        step_id=0,
        env_name="alfworld/AlfredTWEnv",
        task_text="put an apple in the fridge",
        anchor_obs="You see an apple.",
        raw_response="<think>bad</think><action>fly away</action>",
        parsed_action=parse_response("<think>bad</think><action>fly away</action>", "alfworld/AlfredTWEnv"),
        next_obs="Nothing happens.",
        reward=0.0,
        done=False,
        info={"admissible_commands": ["pick up apple"], "won": False},
        is_action_valid=False,
    )
    graph = build_flow_graph([record], "alfworld/AlfredTWEnv")
    result = compute_step_credits(graph, uid="u1")
    assert result.credits[0].break_penalty > 0
    assert result.credits[0].flow_reward < 0
