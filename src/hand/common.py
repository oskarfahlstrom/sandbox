"""Some functionality."""
from matplotlib import colors


def get_bgr_color(name: str):
    """Convert a named color to BGR format."""
    return tuple(int(c * 255) for c in reversed(colors.to_rgb(name)))

def adjust_bgr_color(color: str, depth_shade: int):
    """Adjust each channel and clamp between 0 and 255."""
    return tuple(max(0, min(255, c - depth_shade)) for c in color)
