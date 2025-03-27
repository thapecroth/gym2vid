import cv2
import pickle
import numpy as np
import subprocess
import os
from typing import Dict, List, Any, Union, Tuple


def create_annotated_video(
    mp4_path: str, 
    pkl_path: str, 
    output_path: str,
) -> None:
    """Create an annotated video with state and action information overlaid.
    
    This function reads a video file and its corresponding pickle file containing
    state and action data, then creates a new video with this information overlaid
    in a navigation bar at the top of each frame.
    
    Args:
        mp4_path: Path to the input MP4 video file
        pkl_path: Path to the pickle file containing states and actions
        output_path: Path where the annotated video will be saved
        
    Raises:
        RuntimeError: If video file can't be opened or written
        ValueError: If data in pickle file is invalid
    """
    # Load the pickle file containing states and actions
    try:
        with open(pkl_path, "rb") as f:
            data: Dict[str, List[Any]] = pickle.load(f)

        if not isinstance(data, dict):
            raise ValueError("Pickle file must contain a dictionary")
        if "states_ls" not in data or "action_ls" not in data:
            raise ValueError("Pickle file missing required keys: states_ls, action_ls")

        states_ls: List[np.ndarray] = data["states_ls"]
        action_ls: List[Union[int, np.ndarray]] = data["action_ls"]

        if len(states_ls) != len(action_ls):
            raise ValueError("States and actions lists must have same length")

    except Exception as e:
        raise RuntimeError(f"Failed to load pickle file {pkl_path}: {str(e)}")

    # Open the video file
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {mp4_path}")

    try:
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        nav_height = 60
        new_height = height + nav_height

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, new_height))
        if not out.isOpened():
            raise RuntimeError(f"Failed to create output video: {output_path}")

        frame_idx = 0
        while cap.isOpened() and frame_idx < len(states_ls):
            ret, frame = cap.read()
            if not ret:
                break

            # Create a new canvas with additional navbar area on top
            canvas = np.zeros((new_height, width, 3), dtype=frame.dtype)
            # Place the original frame into the bottom part of the canvas
            canvas[nav_height:new_height, 0:width] = frame

            # Add text overlays
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (255, 255, 255)  # White color
            line_type = 2

            # Format state and action text
            state_text = f"State: {np.round(states_ls[frame_idx], 3)}"
            action_text = f"Action: {action_ls[frame_idx]}"

            # Draw text in the navbar area on the canvas
            cv2.putText(
                canvas, state_text, (15, 30), font, font_scale, font_color, line_type
            )
            cv2.putText(
                canvas, action_text, (15, 50), font, font_scale, font_color, line_type
            )

            # Write the composite frame with navbar to the output video
            out.write(canvas)
            frame_idx += 1

    finally:
        # Release everything even if there's an error
        cap.release()
        out.release()

    # Verify the output video was created successfully
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"Failed to create output video: {output_path}")


def create_slowed_video(
    input_path: str,
    output_path: str,
    slow_factor: float,
) -> None:
    """Create a slowed-down version of a video.
    
    Args:
        input_path: Path to the input video file
        output_path: Path where the slowed video will be saved
        slow_factor: Factor by which to slow down the video
        
    Raises:
        RuntimeError: If ffmpeg command fails or output video is not created
    """
    if not os.path.exists(input_path):
        raise RuntimeError(f"Input video not found: {input_path}")

    if slow_factor <= 0:
        raise ValueError("Slow factor must be positive")

    try:
        cmd = [
            "ffmpeg",
            "-i",
            input_path,
            "-filter:v",
            f"setpts={slow_factor}*PTS",
            "-y",  # Overwrite output file if it exists
            output_path,
        ]
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg command failed: {e.stderr}")

    # Verify the output video was created
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"Failed to create slowed video: {output_path}")


if __name__ == "__main__":
    # Example usage
    SLOW_FACTOR = 48.0
    ENV_NAME = "FrozenLake-v1"
    base = f"./videos/{ENV_NAME}/"
    mp4_path = f"{ENV_NAME}_episodes_0.mp4"
    pkl_path = f"{ENV_NAME}_episodes_0.pkl"
    output_path = f"{ENV_NAME}_episodes_0_annotated.mp4"

    create_annotated_video(base + mp4_path, base + pkl_path, base + output_path)

    slower_output_path = f"{base}{ENV_NAME}_episodes_0_annotated_slowed_{SLOW_FACTOR}x.mp4"
    create_slowed_video(
        base + f"{ENV_NAME}_episodes_0_annotated.mp4",
        slower_output_path,
        SLOW_FACTOR,
    )
    print(f"Created slowed video: {slower_output_path}")
