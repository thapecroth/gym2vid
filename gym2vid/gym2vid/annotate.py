import cv2
import pickle
import numpy as np
import subprocess


def create_annotated_video(mp4_path: str, pkl_path: str, output_path: str):
    # Load the pickle file containing states and actions
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    states_ls = data["states_ls"]
    action_ls = data["action_ls"]

    # Open the video file
    cap = cv2.VideoCapture(mp4_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    nav_height = 60
    new_height = height + nav_height

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, new_height))

    frame_idx = 0
    while cap.isOpened():
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

    # Release everything
    cap.release()
    out.release()


if __name__ == "__main__":
    # Example usage
    SLOW_FACTOR = 48.0
    ENV_NAME = "FrozenLake-v1"
    base = f"./videos/{ENV_NAME}/"
    mp4_path = f"{ENV_NAME}_episodes_0.mp4"
    pkl_path = f"{ENV_NAME}_episodes_0.pkl"
    output_path = f"{ENV_NAME}_episodes_0_annotated.mp4"

    create_annotated_video(base + mp4_path, base + pkl_path, base + output_path)

    slower_output_path = base + f"{ENV_NAME}_episodes_0_annotated_slowed.mp4"
    cmd = [
        "ffmpeg",
        "-i",
        base + f"{ENV_NAME}_episodes_0_annotated.mp4",
        "-filter:v",
        f"setpts={SLOW_FACTOR}*PTS",
        slower_output_path,
    ]
    subprocess.run(cmd, check=True)
    print(f"Created slower video: {slower_output_path}")
