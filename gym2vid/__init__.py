"""gym2vid: A package for training RL agents and recording gameplay videos with annotations."""

from gym2vid.gym2vid.runner import Runner
from gym2vid.gym2vid.annotate import create_annotated_video, create_slowed_video

__version__ = "0.1.0"
__all__ = ["Runner", "create_annotated_video", "create_slowed_video"] 