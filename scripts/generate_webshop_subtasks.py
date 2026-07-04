#!/usr/bin/env python3
import argparse
import hashlib
import itertools
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from obli_flow.alfworld_expert import task_key  # noqa: E402
from obli_flow.llm_obligations import try_build_llm_obligations  # noqa: E402


DEFAULT_OUTPUT = "/opt/tiger/webshop_subtasks.jsonl"
DEFAULT_MODEL = "gpt-5.5-2026-04-24"
DEFAULT_WORKERS = 20
ENV_NAME = "webshop/WebAgentTextEnv"
PRICE_RANGE = [10.0 * i for i in range(1, 100)]


def main() -> int:
    args = parse_args()
    file_path, attr_path, human_attr_path = _resolve_data_paths(args)
    goals = build_webshop_goals(
        file_path=file_path,
        attr_path=attr_path,
        human_attr_path=human_attr_path,
        seed=args.seed,
        human_goals=args.human_goals,
        num_products=args.num_products,
    )
    goals = _select_split(goals, args.split)

    if args.limit is not None:
        goals = goals[args.start : args.start + args.limit]
    elif args.start:
        goals = goals[args.start :]

    if args.dry_run:
        print(f"Found {len(goals)} WebShop goals.")
        for goal in goals[: min(5, len(goals))]:
            print(json.dumps(_dry_run_payload(goal), ensure_ascii=False))
        return 0

    output = Path(args.output)
    errors_output = Path(args.errors_output or f"{args.output}.errors")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        output.write_text("", encoding="utf-8")
        errors_output.write_text("", encoding="utf-8")

    done = _loaded_sources(output) if args.resume and output.exists() else set()
    config = _llm_config(args)
    pending = [goal for goal in goals if goal["source_uid"] not in done]
    generated = 0
    skipped = len(goals) - len(pending)
    failed = 0

    with open(output, "a", encoding="utf-8") as out_f, open(errors_output, "a", encoding="utf-8") as err_f:
        print(f"Processing {len(pending)} pending goals with workers={args.workers}; skipped_existing={skipped}", flush=True)
        progress = _ProgressBar(total=len(pending), label="WebShop subtasks", enabled=args.progress)
        try:
            if args.workers <= 1:
                for completed, goal in enumerate(pending, 1):
                    previous_failed = failed
                    generated, failed = _process_one_serial(
                        goal=goal,
                        total=len(goals),
                        config=config,
                        out_f=out_f,
                        err_f=err_f,
                        generated=generated,
                        failed=failed,
                        strict=args.strict,
                    )
                    progress.update(
                        completed=completed,
                        generated=generated,
                        failed=failed,
                        current=f"goal {goal['goal_index'] + 1}/{len(goals)}",
                        force=failed > previous_failed,
                    )
            else:
                generated, failed = _process_parallel(
                    pending=pending,
                    total=len(goals),
                    workers=args.workers,
                    config=config,
                    out_f=out_f,
                    err_f=err_f,
                    generated=generated,
                    failed=failed,
                    strict=args.strict,
                    progress=progress,
                )
        finally:
            progress.finish(generated=generated, failed=failed)

    print(
        json.dumps(
            {
                "output": str(output),
                "errors_output": str(errors_output),
                "total_selected": len(goals),
                "generated": generated,
                "skipped_existing": skipped,
                "failed": failed,
                "model": args.model,
                "workers": args.workers,
                "seed": args.seed,
                "split": args.split,
                "human_goals": args.human_goals,
                "file_path": str(file_path),
                "attr_path": str(attr_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if failed == 0 or not args.strict else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate offline WebShop subtasks with GPT-5.5.")
    parser.add_argument("--file-path", default=None, help="Path to items_shuffle*.json.")
    parser.add_argument("--attr-path", default=None, help="Path to items_ins*.json.")
    parser.add_argument("--human-attr-path", default=None, help="Path to items_human_ins.json.")
    parser.add_argument("--use-small", action=argparse.BooleanOptionalAction, default=True, help="Use the 1k WebShop item files.")
    parser.add_argument("--human-goals", action=argparse.BooleanOptionalAction, default=False, help="Use human goals instead of synthetic goals.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used by WebShop goal generation and shuffling.")
    parser.add_argument("--split", choices=["all", "train", "eval"], default="all", help="Goal split after WebShop shuffle.")
    parser.add_argument("--num-products", type=int, default=None, help="Optional product limit passed before goal generation.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL path.")
    parser.add_argument("--errors-output", default=None, help="Error JSONL path. Defaults to OUTPUT.errors.")
    parser.add_argument("--model", default=os.environ.get("GLM_MODEL", DEFAULT_MODEL), help="OpenAI-compatible model name.")
    parser.add_argument("--max-subtasks", type=int, default=6)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--cache-dir", default=str(REPO_ROOT / ".cache" / "obliflow_llm"))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Concurrent LLM requests. Use 1 for serial generation.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of goals to process.")
    parser.add_argument("--start", type=int, default=0, help="Optional start offset after sorting/filtering goals.")
    parser.add_argument("--dry-run", action="store_true", help="List selected goals without calling the LLM.")
    parser.add_argument("--overwrite", action="store_true", help="Clear output files before generating.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Do not skip source UIDs already in output.")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True, help="Show a progress bar while generating.")
    parser.add_argument("--strict", action="store_true", help="Abort on the first failed goal.")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def build_webshop_goals(
    *,
    file_path: Path,
    attr_path: Path,
    human_attr_path: Path,
    seed: int,
    human_goals: bool,
    num_products: int | None,
) -> list[dict[str, Any]]:
    random.seed(seed)
    products = _load_products(
        file_path=file_path,
        attr_path=attr_path,
        human_attr_path=human_attr_path,
        human_goals=human_goals,
        num_products=num_products,
    )
    product_prices = _generate_product_prices(products)
    if human_goals:
        goals = _build_human_goals(products, product_prices)
    else:
        goals = _build_synthetic_goals(products, product_prices)

    random.seed(seed)
    random.shuffle(goals)
    for goal_index, goal in enumerate(goals):
        goal["goal_index"] = goal_index
        goal["split"] = "eval" if goal_index < 500 else "train"
        goal["source_uid"] = _goal_uid(goal, seed=seed, human_goals=human_goals)
        goal["task_key"] = task_key(goal["instruction_text"])
    return goals


def _load_products(
    *,
    file_path: Path,
    attr_path: Path,
    human_attr_path: Path,
    human_goals: bool,
    num_products: int | None,
) -> list[dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        raw_products = json.load(f)
    with open(attr_path, "r", encoding="utf-8") as f:
        attributes = json.load(f)
    with open(human_attr_path, "r", encoding="utf-8") as f:
        human_attributes = json.load(f)

    if num_products is not None:
        raw_products = raw_products[:num_products]

    asins: set[str] = set()
    products: list[dict[str, Any]] = []
    for raw in raw_products:
        asin = str(raw.get("asin", ""))
        if asin == "nan" or len(asin) > 10 or asin in asins:
            continue
        asins.add(asin)

        product = dict(raw)
        product["category"] = raw.get("category", "")
        product["query"] = str(raw.get("query", "")).lower().strip()
        product["product_category"] = raw.get("product_category", "")
        product["Title"] = raw.get("name", "")
        product["Description"] = raw.get("full_description", "")
        product["BulletPoints"] = raw.get("small_description") if isinstance(raw.get("small_description"), list) else [raw.get("small_description")]
        product["options"] = _customization_options(raw.get("customization_options") or {})

        attr_entry = attributes.get(asin, {}) if isinstance(attributes, dict) else {}
        product["Attributes"] = attr_entry.get("attributes") or ["DUMMY_ATTR"]
        if human_goals:
            if asin in human_attributes:
                product["instructions"] = human_attributes[asin]
        else:
            product["instruction_text"] = attr_entry.get("instruction")
            product["instruction_attributes"] = attr_entry.get("instruction_attributes")
        products.append(product)
    return products


def _customization_options(raw_options: dict[str, Any]) -> dict[str, list[str]]:
    options: dict[str, list[str]] = {}
    for option_name, option_contents in raw_options.items():
        if option_contents is None:
            continue
        values = []
        for option_content in option_contents:
            value = str(option_content.get("value", "")).strip().replace("/", " | ").lower()
            if value:
                values.append(value)
        if values:
            options[str(option_name).lower()] = values
    return options


def _generate_product_prices(products: list[dict[str, Any]]) -> dict[str, float]:
    prices: dict[str, float] = {}
    for product in products:
        pricing = _parse_pricing(product.get("pricing"))
        if not pricing:
            price = 100.0
        elif len(pricing) == 1:
            price = pricing[0]
        else:
            price = random.uniform(*pricing[:2])
        prices[str(product["asin"])] = price
    return prices


def _parse_pricing(value: Any) -> list[float]:
    if value is None or value == "":
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        return [float(item) for item in value if _is_number(item)]
    prices = []
    for raw in str(value).split("$")[1:]:
        text = re.sub(r"[^\d.]", "", raw)
        if text:
            prices.append(float(Decimal(text)))
    return prices[:2]


def _build_human_goals(products: list[dict[str, Any]], product_prices: dict[str, float]) -> list[dict[str, Any]]:
    goals: list[dict[str, Any]] = []
    for item in products:
        asin = item["asin"]
        if "instructions" not in item:
            continue
        for instruction in item["instructions"]:
            attributes = instruction.get("instruction_attributes") or []
            if not attributes:
                continue
            price_upper, price_text = _price_constraint(product_prices.get(asin))
            goals.append(
                _goal(
                    asin=asin,
                    category=item.get("category"),
                    query=item.get("query"),
                    name=item.get("name"),
                    product_category=item.get("product_category"),
                    instruction_text=str(instruction.get("instruction", "")).strip(".") + price_text,
                    attributes=attributes,
                    price_upper=price_upper,
                    goal_options=instruction.get("instruction_options") or [],
                    weight=1.0,
                )
            )
    return goals


def _build_synthetic_goals(products: list[dict[str, Any]], product_prices: dict[str, float]) -> list[dict[str, Any]]:
    goals: list[dict[str, Any]] = []
    attr_counts: dict[str, int] = {}
    for product in products:
        instruction_text = product.get("instruction_text")
        if not instruction_text:
            continue
        attributes = product.get("instruction_attributes") or []
        if not attributes:
            continue
        price_upper, price_text = _price_constraint(product_prices.get(product["asin"]))

        option_names = sorted(product.get("options") or {})
        combinations = itertools.product(*(product["options"][name] for name in option_names))
        for combination in combinations:
            goal_options = {name: option for name, option in zip(option_names, combination)}
            option_text = ", and ".join([f"{name}: {option}" for name, option in goal_options.items()])
            option_text = " with " + option_text if option_text else ""
            goals.append(
                _goal(
                    asin=product["asin"],
                    category=product.get("category"),
                    query=product.get("query"),
                    name=product.get("Title"),
                    product_category=product.get("product_category"),
                    instruction_text=f"{instruction_text}{option_text}{price_text}",
                    attributes=attributes,
                    price_upper=price_upper,
                    goal_options=goal_options,
                    weight=1.0,
                )
            )
            for attr in attributes:
                attr_counts[attr] = attr_counts.get(attr, 0) + 1

    for goal in goals:
        attributes = goal.get("attributes") or []
        if attributes:
            goal["weight"] = sum(1.0 / attr_counts[attr] for attr in attributes) / len(attributes)
    return goals


def _goal(
    *,
    asin: Any,
    category: Any,
    query: Any,
    name: Any,
    product_category: Any,
    instruction_text: str,
    attributes: Any,
    price_upper: float,
    goal_options: Any,
    weight: float,
) -> dict[str, Any]:
    return {
        "asin": str(asin),
        "category": category,
        "query": query,
        "name": name,
        "product_category": product_category,
        "instruction_text": instruction_text,
        "attributes": attributes,
        "price_upper": price_upper,
        "goal_options": goal_options,
        "weight": weight,
    }


def _price_constraint(price: float | None) -> tuple[float, str]:
    if price is None:
        return 1000000.0, ""
    price_range = [value for value in PRICE_RANGE if value > price][:4]
    if len(price_range) < 2:
        return 1000000.0, ""
    _, price_upper = sorted(random.sample(price_range, 2))
    return price_upper, f", and price lower than {price_upper:.2f} dollars"


def _llm_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "use_llm_decomposition": True,
        "llm_use_expert_trajectory": False,
        "llm_fallback_to_rules": False,
        "llm_model": args.model,
        "llm_max_subtasks": args.max_subtasks,
        "llm_timeout_s": args.timeout_s,
        "llm_max_retries": args.max_retries,
        "llm_temperature": None,
        "llm_max_tokens": args.max_tokens,
        "llm_cache_dir": args.cache_dir,
    }


def _generate_entry(goal: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
    task_text = goal["instruction_text"]
    info = {
        "source_uid": goal["source_uid"],
        "webshop_goal_index": goal["goal_index"],
        "webshop_split": goal["split"],
    }
    obligations = try_build_llm_obligations(
        env_name=ENV_NAME,
        traj_uid=goal["source_uid"],
        task_text=task_text,
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
    _normalize_subtask_weights(subtasks)

    return {
        "version": 1,
        "env_name": ENV_NAME,
        "model": config["llm_model"],
        "source": "webshop_goal",
        "source_uid": goal["source_uid"],
        "goal_index": goal["goal_index"],
        "split": goal["split"],
        "task": task_text,
        "task_key": goal["task_key"],
        "goal_reference": _goal_reference(goal),
        "subtasks": subtasks,
    }


def _normalize_subtask_weights(subtasks: list[dict[str, Any]]) -> None:
    weights: list[float] = []
    for subtask in subtasks:
        try:
            weight = float(subtask.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        weights.append(max(weight, 0.0))

    total = sum(weights)
    if total <= 0.0:
        normalized = [1.0 / len(subtasks)] * len(subtasks)
    else:
        normalized = [weight / total for weight in weights]

    for subtask, weight in zip(subtasks, normalized):
        subtask["weight"] = weight


def _generate_from_goal(goal: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
    entry = _generate_entry(goal, config=config)
    return {
        "goal_index": goal["goal_index"],
        "task": goal["instruction_text"],
        "entry": entry,
    }


def _process_one_serial(
    *,
    goal: dict[str, Any],
    total: int,
    config: dict[str, Any],
    out_f,
    err_f,
    generated: int,
    failed: int,
    strict: bool,
) -> tuple[int, int]:
    try:
        entry = _generate_entry(goal, config=config)
        _write_jsonl(out_f, entry)
        return generated + 1, failed
    except Exception as exc:
        _write_error(err_f, goal, exc)
        if strict:
            raise
        return generated, failed + 1


def _process_parallel(
    *,
    pending: list[dict[str, Any]],
    total: int,
    workers: int,
    config: dict[str, Any],
    out_f,
    err_f,
    generated: int,
    failed: int,
    strict: bool,
    progress: "_ProgressBar",
) -> tuple[int, int]:
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_to_goal = {
            executor.submit(_generate_from_goal, goal, config=config): goal
            for goal in pending
        }
        for future in as_completed(future_to_goal):
            goal = future_to_goal[future]
            completed += 1
            try:
                result = future.result()
                _write_jsonl(out_f, result["entry"])
                generated += 1
                progress.update(
                    completed=completed,
                    generated=generated,
                    failed=failed,
                    current=f"goal {result['goal_index'] + 1}/{total}",
                )
            except Exception as exc:
                failed += 1
                _write_error(err_f, goal, exc)
                progress.update(
                    completed=completed,
                    generated=generated,
                    failed=failed,
                    current=f"goal {goal['goal_index'] + 1}/{total}",
                    force=True,
                )
                if strict:
                    raise
    return generated, failed


class _ProgressBar:
    def __init__(self, *, total: int, label: str, enabled: bool, width: int = 28):
        self.total = max(0, total)
        self.label = label
        self.enabled = enabled
        self.width = width
        self.started_at = time.monotonic()
        self.last_render_at = 0.0
        self.last_line_len = 0
        self.completed = 0
        if self.enabled:
            self.update(completed=0, generated=0, failed=0, force=True)

    def update(
        self,
        *,
        completed: int,
        generated: int,
        failed: int,
        current: str = "",
        force: bool = False,
    ) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        self.completed = min(max(0, completed), self.total)
        if not force and self.completed < self.total and now - self.last_render_at < 0.25:
            return
        self.last_render_at = now

        fraction = 1.0 if self.total == 0 else self.completed / self.total
        filled = int(round(self.width * fraction))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = max(0.0, now - self.started_at)
        rate = self.completed / elapsed if elapsed > 0.0 else 0.0
        remaining = self.total - self.completed
        eta = _format_duration(remaining / rate) if rate > 0.0 and remaining > 0 else "0s"
        line = (
            f"\r{self.label} [{bar}] {self.completed}/{self.total} "
            f"{fraction * 100:5.1f}% ok={generated} fail={failed} "
            f"rate={rate:.2f}/s eta={eta}"
        )
        if current:
            line += f" {current}"

        padding = " " * max(0, self.last_line_len - len(line))
        sys.stderr.write(line + padding)
        sys.stderr.flush()
        self.last_line_len = len(line)

    def finish(self, *, generated: int, failed: int) -> None:
        if not self.enabled:
            return
        self.update(
            completed=self.total,
            generated=generated,
            failed=failed,
            current="done",
            force=True,
        )
        sys.stderr.write("\n")
        sys.stderr.flush()


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _write_jsonl(handle, entry: dict[str, Any]) -> None:
    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    handle.flush()


def _write_error(handle, goal: dict[str, Any], exc: Exception) -> None:
    error_entry = {
        "source_uid": goal.get("source_uid"),
        "goal_index": goal.get("goal_index"),
        "task": goal.get("instruction_text"),
        "error": str(exc),
    }
    handle.write(json.dumps(error_entry, ensure_ascii=False) + "\n")
    handle.flush()
    print(f"ERROR goal={goal.get('goal_index')} uid={goal.get('source_uid')}: {exc}", file=sys.stderr, flush=True)


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
            if entry.get("source_uid"):
                sources.add(str(entry["source_uid"]))
    return sources


def _resolve_data_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    data_dir = _find_webshop_data_dir()
    file_name = "items_shuffle_1000.json" if args.use_small else "items_shuffle.json"
    attr_name = "items_ins_v2_1000.json" if args.use_small else "items_ins_v2.json"
    file_path = Path(args.file_path) if args.file_path else data_dir / file_name
    attr_path = Path(args.attr_path) if args.attr_path else data_dir / attr_name
    human_attr_path = Path(args.human_attr_path) if args.human_attr_path else data_dir / "items_human_ins.json"
    for path in (file_path, attr_path, human_attr_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing WebShop data file: {path}")
    return file_path, attr_path, human_attr_path


def _find_webshop_data_dir() -> Path:
    candidates = [
        REPO_ROOT / "agent_system/environments/env_package/webshop/webshop/data",
        Path("/opt/tiger/obliflow-grpo/agent_system/environments/env_package/webshop/webshop/data"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _select_split(goals: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split == "all":
        return goals
    return [goal for goal in goals if goal.get("split") == split]


def _goal_uid(goal: dict[str, Any], *, seed: int, human_goals: bool) -> str:
    payload = {
        "dataset": "webshop",
        "seed": seed,
        "human_goals": human_goals,
        "goal_index": goal.get("goal_index"),
        "asin": goal.get("asin"),
        "task": goal.get("instruction_text"),
        "attributes": goal.get("attributes"),
        "goal_options": goal.get("goal_options"),
        "price_upper": goal.get("price_upper"),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _goal_reference(goal: dict[str, Any]) -> dict[str, Any]:
    return {
        "asin": goal.get("asin"),
        "query": goal.get("query"),
        "name": goal.get("name"),
        "category": goal.get("category"),
        "product_category": goal.get("product_category"),
        "attributes": goal.get("attributes"),
        "goal_options": goal.get("goal_options"),
        "price_upper": goal.get("price_upper"),
        "goal_weight": goal.get("weight"),
    }


def _dry_run_payload(goal: dict[str, Any]) -> dict[str, Any]:
    return {
        "goal_index": goal.get("goal_index"),
        "split": goal.get("split"),
        "source_uid": goal.get("source_uid"),
        "task": goal.get("instruction_text"),
        "goal_reference": _goal_reference(goal),
    }


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
