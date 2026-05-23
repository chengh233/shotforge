"""Export the final frame of a clip as an image.

    python -m tools.last_frame --video projects/example/out/s1.mp4 \
                               --out   projects/example/frames/s1b.png

Use it to chain a longer continuous take: the last frame of one sub-shot
becomes the starting frame of the next.
"""
from __future__ import annotations

import argparse

import imageio.v3 as iio


def main() -> None:
    parser = argparse.ArgumentParser(description="Save a video's last frame as an image.")
    parser.add_argument("--video", required=True, help="input mp4")
    parser.add_argument("--out", required=True, help="output image (e.g. frames/s2.png)")
    args = parser.parse_args()

    # pyav decodes the clip into a stack of frames; take the last one.
    frames = iio.imread(args.video, plugin="pyav")
    iio.imwrite(args.out, frames[-1])
    print(f"[ok] {args.video} last frame -> {args.out}")


if __name__ == "__main__":
    main()
