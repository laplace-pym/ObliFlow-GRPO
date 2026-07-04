#!/usr/bin/env python3
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from obli_flow.alfworld_expert import (  # noqa: E402
    DEFAULT_ALFWORLD_ROOT,
    compact_expert_for_prompt,
    iter_alfworld_traj_data,
    load_alfworld_expert,
)
from obli_flow.llm_obligations import try_build_llm_obligations  # noqa: E402


DEFAULT_OUTPUT = "/opt/tiger/alfworld_subtasks.jsonl"
DEFAULT_MODEL = "gpt-5.5-2026-04-24"
DEFAULT_WORKERS = 20
ENV_NAME = "alfworld/AlfredTWEnv"


def main() -> int:
    args = parse_args()
    paths = iter_alfworld_traj_data(args.alfworld_root, splits=args.splits)
    if args.limit is not None:
        paths = paths[args.start : args.start + args.limit]
    elif args.start:
        paths = paths[args.start :]

    if args.dry_run:
        print(f"Found {len(paths)} ALFWorld traj_data.json files.")
        for path in paths[: min(5, len(paths))]:
            expert = load_alfworld_expert(path)
            print(json.dumps({"task": expert["task"], "task_type": expert["task_type"], "source_path": str(path)}, ensure_ascii=False))
        return 0

    output = Path(args.output)
    errors_output = Path(args.errors_output or f"{args.output}.errors")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        output.write_text("", encoding="utf-8")
        errors_output.write_text("", encoding="utf-8")

    done = _loaded_sources(output) if args.resume and output.exists() else set()
    config = _llm_config(args)
    pending = [(index, path) for index, path in enumerate(paths, 1) if str(path) not in done]
    generated = 0
    skipped = len(paths) - len(pending)
    failed = 0

    with open(output, "a", encoding="utf-8") as out_f, open(errors_output, "a", encoding="utf-8") as err_f:
        print(f"Processing {len(pending)} pending cases with workers={args.workers}; skipped_existing={skipped}", flush=True)
        if args.workers <= 1:
            for index, path in pending:
                generated, failed = _process_one_serial(
                    index=index,
                    total=len(paths),
                    path=path,
                    config=config,
                    out_f=out_f,
                    err_f=err_f,
                    generated=generated,
                    failed=failed,
                    strict=args.strict,
                )
        else:
            generated, failed = _process_parallel(
                pending=pending,
                total=len(paths),
                workers=args.workers,
                config=config,
                out_f=out_f,
                err_f=err_f,
                generated=generated,
                failed=failed,
                strict=args.strict,
            )

    print(
        json.dumps(
            {
                "output": str(output),
                "errors_output": str(errors_output),
                "total_selected": len(paths),
                "generated": generated,
                "skipped_existing": skipped,
                "failed": failed,
                "model": args.model,
                "workers": args.workers,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if failed == 0 or not args.strict else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate offline expert-assisted ALFWorld subtasks with GPT-5.5.")
    parser.add_argument("--alfworld-root", default=DEFAULT_ALFWORLD_ROOT, help="Root containing ALFWorld json_2.1.1 splits.")
    parser.add_argument("--splits", nargs="*", default=None, help="Optional split list, e.g. train valid_seen valid_unseen.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL path.")
    parser.add_argument("--errors-output", default=None, help="Error JSONL path. Defaults to OUTPUT.errors.")
    parser.add_argument("--model", default=os.environ.get("GLM_MODEL", DEFAULT_MODEL), help="OpenAI-compatible model name.")
    parser.add_argument("--max-subtasks", type=int, default=6)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--cache-dir", default=str(REPO_ROOT / ".cache" / "obliflow_llm"))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Concurrent LLM requests. Use 1 for serial generation.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of cases to process.")
    parser.add_argument("--start", type=int, default=0, help="Optional start offset after sorting files.")
    parser.add_argument("--dry-run", action="store_true", help="List selected cases without calling the LLM.")
    parser.add_argument("--overwrite", action="store_true", help="Clear output files before generating.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Do not skip source paths already in output.")
    parser.add_argument("--strict", action="store_true", help="Abort on the first failed case.")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def _llm_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "use_llm_decomposition": True,
        "llm_use_expert_trajectory": True,
        "llm_fallback_to_rules": False,
        "llm_model": args.model,
        "llm_max_subtasks": args.max_subtasks,
        "llm_timeout_s": args.timeout_s,
        "llm_max_retries": args.max_retries,
        "llm_temperature": None,
        "llm_max_tokens": args.max_tokens,
        "llm_cache_dir": args.cache_dir,
    }


def _generate_entry(expert: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
    info = {
        "expert": expert,
        "traj_data_path": expert["source_path"],
        "extra.gamefile": expert["gamefile"],
    }
    obligations = try_build_llm_obligations(
        env_name=ENV_NAME,
        traj_uid=expert["source_uid"],
        task_text=expert["task"],
        info=info,
        config=config,
    )
    if not obligations:
        raise RuntimeError("LLM returned no subtasks")

    subtasks = []
    for obligation in obligations:
        if obligation.type != "llm_subtask":
            continue
        subtasks.append(
            {
                "id": obligation.target.get("subtask_id"),
                "description": obligation.target.get("description"),
                "success_criteria": obligation.target.get("success_criteria"),
                "weight": obligation.weight,
                "order": obligation.metadata.get("order"),
            }
        )
    if not subtasks:
        raise RuntimeError("LLM returned obligations but no llm_subtask entries")

    return {
        "version": 1,
        "env_name": ENV_NAME,
        "model": config["llm_model"],
        "source_path": expert["source_path"],
        "source_uid": expert["source_uid"],
        "gamefile": expert["gamefile"],
        "split": expert.get("split"),
        "task_type": expert.get("task_type"),
        "task": expert["task"],
        "task_key": expert["task_key"],
        "expert_reference": compact_expert_for_prompt(expert),
        "subtasks": subtasks,
    }


def _generate_from_path(index: int, path: Path, *, config: dict[str, Any]) -> dict[str, Any]:
    expert = load_alfworld_expert(path)
    entry = _generate_entry(expert, config=config)
    return {
        "index": index,
        "source_path": str(path),
        "task_type": expert.get("task_type"),
        "task": expert.get("task"),
        "entry": entry,
    }


def _process_one_serial(
    *,
    index: int,
    total: int,
    path: Path,
    config: dict[str, Any],
    out_f,
    err_f,
    generated: int,
    failed: int,
    strict: bool,
) -> tuple[int, int]:
    try:
        expert = load_alfworld_expert(path)
        print(f"[{index}/{total}] decompose {expert['task_type']} :: {expert['task']}", flush=True)
        entry = _generate_entry(expert, config=config)
        _write_jsonl(out_f, entry)
        return generated + 1, failed
    except Exception as exc:
        _write_error(err_f, path, exc)
        if strict:
            raise
        return generated, failed + 1


def _process_parallel(
    *,
    pending: list[tuple[int, Path]],
    total: int,
    workers: int,
    config: dict[str, Any],
    out_f,
    err_f,
    generated: int,
    failed: int,
    strict: bool,
) -> tuple[int, int]:
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_to_path = {
            executor.submit(_generate_from_path, index, path, config=config): (index, path)
            for index, path in pending
        }
        for future in as_completed(future_to_path):
            index, path = future_to_path[future]
            completed += 1
            try:
                result = future.result()
                _write_jsonl(out_f, result["entry"])
                generated += 1
                print(
                    f"[done {completed}/{len(pending)} | case {index}/{total}] "
                    f"{result['task_type']} :: {result['task']}",
                    flush=True,
                )
            except Exception as exc:
                failed += 1
                _write_error(err_f, path, exc)
                if strict:
                    raise
    return generated, failed


def _write_jsonl(handle, entry: dict[str, Any]) -> None:
    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    handle.flush()


def _write_error(handle, path: Path, exc: Exception) -> None:
    error_entry = {"source_path": str(path), "error": str(exc)}
    handle.write(json.dumps(error_entry, ensure_ascii=False) + "\n")
    handle.flush()
    print(f"ERROR {path}: {exc}", file=sys.stderr, flush=True)


def _loaded_sources(path: Path) -> set[str]:
    sources: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            if entry.get("source_path"):
                sources.add(str(entry["source_path"]))
    return sources


if __name__ == "__main__":
    raise SystemExit(main())
