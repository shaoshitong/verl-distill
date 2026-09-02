import math
from typing import Optional, Tuple


def normalize_exit_point_choices(
    exit_point_choices,
    num_student_steps: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    num_student_steps = int(num_student_steps)
    if exit_point_choices is None:
        points = tuple(range(num_student_steps, 0, -1))
    elif isinstance(exit_point_choices, str):
        stripped = exit_point_choices.strip().strip("[]")
        points = tuple(int(part.strip()) for part in stripped.split(",") if part.strip())
    else:
        points = tuple(int(value) for value in exit_point_choices)

    if not points:
        raise ValueError("exit_point_choices must not be empty")
    if len(set(points)) != len(points):
        raise ValueError("exit_point_choices must not contain duplicates")
    invalid = [value for value in points if value < 1 or value > num_student_steps]
    if invalid:
        raise ValueError(
            f"exit_point_choices values must be in [1, {num_student_steps}], got {invalid}"
        )

    allowed_exit_steps = tuple(num_student_steps - point + 1 for point in points)
    return points, allowed_exit_steps


def normalize_exit_point_weights(
    exit_point_weights,
    exit_point_choices: Tuple[int, ...],
) -> Optional[Tuple[float, ...]]:
    if exit_point_weights is None:
        return None
    if isinstance(exit_point_weights, str):
        stripped = exit_point_weights.strip().strip("[]")
        weights = tuple(float(part.strip()) for part in stripped.split(",") if part.strip())
    else:
        weights = tuple(float(value) for value in exit_point_weights)
    if len(weights) != len(exit_point_choices):
        raise ValueError("exit_point_weights must have the same length as exit_point_choices")
    if any(not math.isfinite(weight) for weight in weights):
        raise ValueError("exit_point_weights must be finite")
    if any(weight < 0.0 for weight in weights):
        raise ValueError("exit_point_weights must be non-negative")
    if sum(weights) <= 0.0:
        raise ValueError("exit_point_weights must contain a positive weight")
    return weights
