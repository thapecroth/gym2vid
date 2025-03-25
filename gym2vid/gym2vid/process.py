import fire
from glob import glob
import subprocess
from pathlib import Path
import os
import random
import string
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple
from tqdm import tqdm
import shutil


# Set CUDA_VISIBLE_DEVICES to use only the first GPU (index 0)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

WIDTH = 640
HEIGHT = 360  # 240
N_GPU = 3
GPU_OFFSET = 0
BATCH_SIZE = 500
PARALLEL = True


def tokenize_video(args: Tuple[int, int]):
    bucket_idx, bucket_size = args
    env = os.environ.copy()
    print(f"tokenizing video at bucket {bucket_idx}", flush=True)
    # env.update(
    #     {
    #         "CUDA_VISIBLE_DEVICES": str(GPU_OFFSET + (bucket_idx % N_GPU)),
    #     }
    # )

    # CUDA_VISIBLE_DEVICES=0 python varicb_tokenize_video_bucket.py --data_path_jsonl ./varicb_bucket_paths.jsonl --bucket_size 5 --bucket_idx 0 --model_path_i ./checkpoints/cvpr2024_image.pth.tar --model_path_p ./checkpoints/cvpr2024_video.pth.tar --rate_num 1 --q_indexes_i 32 --cuda 1 --float16 1 --worker 1 --encode_mode_dir ./video_bucket_processed
    commands = [
        "python",
        "varicb_tokenize_video_bucket.py",
        "--data_path_jsonl",
        str(Path("./varicb_bucket_paths.jsonl").absolute()),
        "--bucket_size",
        str(bucket_size),
        "--bucket_idx",
        str(bucket_idx),
        "--model_path_i",
        "./checkpoints/cvpr2024_image.pth.tar",
        "--model_path_p",
        "./checkpoints/cvpr2024_video.pth.tar",
        "--rate_num",
        "1",
        "--q_indexes_i",
        "32",
        "--cuda",
        "1",
        "--float16",
        "1",
        "--worker",
        str(os.cpu_count()),
        "--cuda_idx",
        str(GPU_OFFSET + (bucket_idx % N_GPU)),
        "--encode_mode_dir",
        "./video_bucket_processed",
    ]
    commands_str = " ".join(commands)
    print(f"tokenizing video with {commands_str}", flush=True)
    process = subprocess.run(
        commands,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        check=True,
    )

    print(process.stdout)

    subprocess.run(["rm", "-rf", "/tmp/tmp*"])


def detokenized_to_video(bucket_idx: int):
    env = os.environ.copy()
    # env.update(
    #     {
    #         "CUDA_VISIBLE_DEVICES": str(GPU_OFFSET + (bucket_idx % N_GPU)),
    #     }
    # )
    subprocess.run(
        [
            "python",
            "varicb_tokenize_video_bucket.py",
            "--data_path_jsonl",
            "placeholder",
            "--bucket_size",
            "-1",
            "--bucket_idx",
            str(bucket_idx),
            "--model_path_i",
            "./checkpoints/cvpr2024_image.pth.tar",
            "--model_path_p",
            "./checkpoints/cvpr2024_video.pth.tar",
            "--rate_num",
            "1",
            "--q_indexes_i",
            "32",
            "--cuda",
            "1",
            "--float16",
            "1",
            "--worker",
            str(os.cpu_count()),
            "--cuda_idx",
            str(GPU_OFFSET + (bucket_idx % N_GPU)),
            "--decode_mode_dir",
            "./video_bucket_processed",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def tensor_to_tokens(args: Tuple[int, str]):
    try:
        bucket_idx, dataset_name = args
        env = os.environ.copy()
        # env.update(
        #     {
        #         "CUDA_VISIBLE_DEVICES": str(bucket_idx % N_GPU),
        #     }
        # )
        # python varicb_tokenized_json_to_seq\ .py --dataset_name="./video_bucket_processed/bucket_0" --export_dataset_path='cartpole-64k' --to_hub
        commands = [
            "python",
            "varicb_tokenized_json_to_seq.py",
            "--dataset_name",
            f"./video_bucket_processed/bucket_{bucket_idx}",
            "--export_dataset_path",
            dataset_name,
            "--to_hub",
        ]
        print(" ".join(commands))
        process = subprocess.run(
            commands,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        subprocess.run(["rm", "-rf", dataset_name])
        print(process.stdout)
    except Exception as e:
        print(f"Error in {args}: {e}")
        if isinstance(e, subprocess.CalledProcessError) and e.stderr:
            print(e.stderr)


def convert_video_into_smaller_parts(path: Path):
    # Obtain video duration using ffprobe
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path.resolve()),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        duration = float(result.stdout.strip())
    except Exception as e:
        print(f"Failed to get duration for {path}: {e}")
        return

    # Skip processing if video is shorter than 5 seconds
    if duration < 5:
        print(
            f"Skipping {path} because its duration {duration:.2f} seconds is less than 5 seconds."
        )
        return

    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(path.resolve()),
            "-c",
            "copy",
            "-map",
            "0",
            "-f",
            "segment",
            "-segment_time",
            "6",
            "-reset_timestamps",
            "1",
            f"./out_tokenization_bin/short_video/{path.stem}_%03d.mp4",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main(input_glob: str, output_dataset_prefix: str, filter_ffmpeg=False):
    og_files = glob(input_glob)
    print(f"OG files length: {len(og_files)}")
    os.makedirs("./out_tokenization_bin", exist_ok=True)
    if Path("./out_tokenization_bin/short_video").exists():
        shutil.rmtree("./out_tokenization_bin/short_video")
    os.makedirs("./out_tokenization_bin/short_video", exist_ok=True)
    cpu_count = os.cpu_count()

    if filter_ffmpeg:
        print("chunking the video into smaller part")
        with ThreadPoolExecutor(
            max_workers=int((cpu_count if cpu_count is not None else 40) * 4)
        ) as executor:
            try:
                results = executor.map(
                    convert_video_into_smaller_parts,
                    [Path(file) for file in og_files],
                )
                for _ in results:
                    pass
            except Exception as e:
                print(e)

        print("finish chunking video into 5s chunks")
        files = glob("./out_tokenization_bin/short_video/*.mp4")
    else:
        files = og_files
    assert len(files) > 0, "empty files"

    with open("./varicb_bucket_paths.jsonl", "w") as f:
        for file in files:
            random_str = "".join(random.choices(string.ascii_letters, k=5))
            f.write(
                f'{{"video_guid": "{random_str}", "video_path": "{file}", "metadata": {{}}}}\n'
            )

    if PARALLEL:
        with ThreadPoolExecutor(max_workers=N_GPU) as executor:
            try:
                results = executor.map(
                    tokenize_video,
                    [
                        tuple((i, BATCH_SIZE))
                        for i in range((len(files) // BATCH_SIZE) + 1)
                    ],
                )
                for _ in results:
                    pass
            except Exception as e:
                print(e)
    else:
        for i in tqdm(range((len(files) // BATCH_SIZE) + 1)):
            try:
                tokenize_video((i, BATCH_SIZE))
            except Exception as e:
                print(e)

    # convert the following command to python subproccess
    # print("converting tokenize back to mp4")
    print("convert tensor to tokens")
    if PARALLEL:
        with ThreadPoolExecutor(max_workers=N_GPU) as executor:
            try:
                results = executor.map(
                    tensor_to_tokens,
                    [
                        tuple((i, f"{output_dataset_prefix}-bucket-{i}"))
                        for i in range((len(files) // BATCH_SIZE) + 1)
                    ],
                )
                for _ in results:
                    pass
            except Exception as e:
                print(e)
    else:
        for i in tqdm(range((len(files) // BATCH_SIZE) + 1)):
            try:
                tensor_to_tokens((i, f"{output_dataset_prefix}-bucket-{i}"))
            except Exception as e:
                print(e)


if __name__ == "__main__":
    fire.Fire(main)
