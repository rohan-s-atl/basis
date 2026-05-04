from __future__ import annotations


def return_bucket(raw_return: float) -> str:
    magnitude = abs(float(raw_return))
    if magnitude < 0.002:
        return "flat"
    if magnitude < 0.01:
        return "small"
    if magnitude < 0.03:
        return "medium"
    return "large"


def benchmark_metrics(
    *,
    predicted_direction: str,
    raw_return: float,
    benchmark_return: float | None,
    noise_threshold: float,
) -> dict[str, float | int | None]:
    if benchmark_return is None:
        return {
            "benchmark_return": None,
            "excess_return": None,
            "benchmark_label": None,
        }

    excess_return = float(raw_return) - float(benchmark_return)
    label = _directional_label(
        predicted_direction,
        excess_return,
        noise_threshold=noise_threshold,
    )
    return {
        "benchmark_return": round(float(benchmark_return), 8),
        "excess_return": round(excess_return, 8),
        "benchmark_label": label,
    }


def directional_label(
    predicted_direction: str,
    raw_return: float,
    *,
    noise_threshold: float,
) -> int | None:
    return _directional_label(
        predicted_direction,
        raw_return,
        noise_threshold=noise_threshold,
    )


def _directional_label(
    predicted_direction: str,
    value: float,
    *,
    noise_threshold: float,
) -> int | None:
    if abs(value) < noise_threshold:
        return None
    if predicted_direction == "up" and value > 0:
        return 1
    if predicted_direction == "down" and value < 0:
        return 1
    return 0
