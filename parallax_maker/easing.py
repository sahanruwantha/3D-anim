# (c) 2024 Niels Provos
#
# Easing functions for smooth camera animations.
#

from enum import Enum
from typing import Callable
import numpy as np


class EasingType(Enum):
    """Available easing function types"""
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    EASE_IN_CUBIC = "ease_in_cubic"
    EASE_OUT_CUBIC = "ease_out_cubic"
    EASE_IN_OUT_CUBIC = "ease_in_out_cubic"


def linear(t: float) -> float:
    """Linear interpolation: f(t) = t"""
    return t


def ease_in_quad(t: float) -> float:
    """Quadratic ease-in: f(t) = t^2"""
    return t * t


def ease_out_quad(t: float) -> float:
    """Quadratic ease-out: f(t) = 1 - (1-t)^2"""
    return 1 - (1 - t) ** 2


def ease_in_out_quad(t: float) -> float:
    """Quadratic ease-in-out"""
    if t < 0.5:
        return 2 * t * t
    return 1 - (-2 * t + 2) ** 2 / 2


def ease_in_cubic(t: float) -> float:
    """Cubic ease-in: f(t) = t^3"""
    return t ** 3


def ease_out_cubic(t: float) -> float:
    """Cubic ease-out: f(t) = 1 - (1-t)^3"""
    return 1 - (1 - t) ** 3


def ease_in_out_cubic(t: float) -> float:
    """Cubic ease-in-out"""
    if t < 0.5:
        return 4 * t ** 3
    return 1 - (-2 * t + 2) ** 3 / 2


EASING_FUNCTIONS: dict[EasingType, Callable[[float], float]] = {
    EasingType.LINEAR: linear,
    EasingType.EASE_IN: ease_in_quad,
    EasingType.EASE_OUT: ease_out_quad,
    EasingType.EASE_IN_OUT: ease_in_out_quad,
    EasingType.EASE_IN_CUBIC: ease_in_cubic,
    EasingType.EASE_OUT_CUBIC: ease_out_cubic,
    EasingType.EASE_IN_OUT_CUBIC: ease_in_out_cubic,
}


def apply_easing(t: float, easing: EasingType) -> float:
    """
    Apply easing function to normalized time t.

    Args:
        t: Normalized time value (0.0 to 1.0)
        easing: The easing type to apply

    Returns:
        Eased value between 0.0 and 1.0
    """
    return EASING_FUNCTIONS[easing](np.clip(t, 0.0, 1.0))
