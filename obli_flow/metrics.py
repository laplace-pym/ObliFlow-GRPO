from collections import defaultdict


def aggregate_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for metrics in metric_dicts:
        for key, value in metrics.items():
            try:
                buckets[key].append(float(value))
            except Exception:
                continue
    return {key: sum(values) / max(1, len(values)) for key, values in buckets.items()}
