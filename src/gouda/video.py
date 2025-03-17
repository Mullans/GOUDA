"""Methods for video processing."""

from __future__ import annotations

import os
from collections.abc import Sequence
from types import TracebackType
from typing import Any

import cv2
import IPython.display
import numpy as np
import numpy.typing as npt

from gouda import data_methods
from gouda.general import extract_method_kwargs


class VideoWriter:
    """A convenience wrapper for OpenCV video writing."""

    def __init__(
        self,
        out_path: str | os.PathLike,
        fps: int = 10,
        codec: str = "MJPG",
        output_shape: tuple[int, int] | None = None,
        interpolator: int = cv2.INTER_LINEAR,
    ) -> None:
        self.out_path = out_path
        self.output_shape = output_shape  # assumes (height, width)
        self.writer: cv2.VideoWriter | None = None
        self.fps = fps
        self.codec = codec
        self.interpolator = interpolator

    def __enter__(self) -> VideoWriter:  # noqa: D105
        return self

    def __exit__(  # noqa: D105
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self.writer is not None:
            self.writer.release()

    def start_writer(self) -> None:
        """Start the video writer."""
        # OpenCV has issues with some type-stubs
        if self.output_shape is None:
            raise ValueError("Output shape must be set before starting the writer")
        codec = cv2.VideoWriter_fourcc(*self.codec)  # type: ignore[attr-defined]
        self.writer = cv2.VideoWriter(str(self.out_path), codec, self.fps, (self.output_shape[1], self.output_shape[0]))

    def write(self, data: npt.NDArray) -> None:
        """Write a frame to the video."""
        if self.writer is None:
            if self.output_shape is None:
                data_shape = np.squeeze(data).shape
                self.output_shape = (data_shape[0], data_shape[1])
            self.start_writer()
        if self.output_shape is None or self.writer is None:
            raise ValueError("Video writer and output shape must be set before writing")
        data = data_methods.to_uint8(data)
        if data.ndim == 2:
            # TODO - allow for color maps
            data = np.dstack([data] * 3)
        elif data.ndim == 3 and data.shape[-1] == 4:
            data = cv2.cvtColor(data, cv2.COLOR_RGBA2RGB)
        if data.shape[:2] != self.output_shape:
            data = cv2.resize(data, (self.output_shape[1], self.output_shape[0]), interpolation=self.interpolator)
        self.writer.write(data)

    def __call__(self, data: npt.NDArray[Any]) -> None:
        """Shortcut to write a frame to the video."""
        self.write(data)


def show_video(
    data: Sequence[npt.ArrayLike] | npt.NDArray,
    player_width: int = 500,
    player_height: int = 300,
    frame_height: int | None = None,
    frame_width: int | None = None,
    file_name: str | os.PathLike = "temp.mp4",
    show: str | None = "ipython",
    **kwargs: Any,  # noqa: ANN401
) -> None | IPython.display.Video:
    """Convert a series of frames to a video and display it.

    Parameters
    ----------
    data : list or numpy.ndarray
        The frames to join into a video
    player_width : int
        The width in pixels of the video player
    player_height : int
        The height in pixels of the video player
    frame_height : int
        The height in pixels of the result video (if None, it will be determined based on the first frame)
    frame_width : int
        The width in pixels of the result video (if None, it will be determined based on the first frame)
    file_name : str or os.PathLike
        The path to save the output video to
    show : str or None
        The method to show the video or None to not display the result (options are 'ipython', 'opencv', None)
    **kwargs : dict
        Other parameters for VideoWriter such as fps, codec, interpolator


    Note
    ----
    Data can be in shape [frames, x, y], [frames, x, y, c], but only 1, 3, or 4 channels will work
    """
    defaults = {
        "fps": 10,
        "codec": "H264",
        "frame_height": frame_height,
        "frame_width": frame_width,
        "interpolator": 1,  # 1 = cv2.InterLinear
    }

    for item, val in defaults.items():
        if item not in kwargs:
            kwargs[item] = val
    if isinstance(data, list | tuple):
        # A list/tuple of arrays
        if hasattr(data[0], "__array__"):
            nframes = len(data)
            temp = np.array(data[0])
            data_shape = temp.shape
            ndim = temp.ndim + 1
        else:
            raise ValueError(f"Frames must be array-like, not {type(data[0])}")
    elif hasattr(data, "__array__"):
        # Array
        data = np.array(data)
        nframes = data.shape[0]
        data_shape = data.shape[1:]
        ndim = data.ndim
    else:
        raise ValueError(f"Unknown data type: {type(data)}")

    if ndim < 2:
        raise ValueError("Video data must have at frames, height, and width")
    if not (ndim == 3 or (ndim == 4 and data_shape[-1] in [1, 3, 4])):
        raise ValueError(f"Unknown video shape: {data_shape}")

    if kwargs["frame_height"] is not None and kwargs["frame_width"] is not None:
        kwargs["output_shape"] = (int(kwargs["frame_height"]), int(kwargs["frame_width"]))
    elif kwargs["frame_height"] is not None:
        width = data_shape[1] * (kwargs["frame_height"] / data_shape[0])
        kwargs["output_shape"] = (int(kwargs["frame_height"]), int(width))
    elif kwargs["frame_width"] is not None:
        height = data_shape[0] * (kwargs["frame_width"] / data_shape[1])
        kwargs["output_shape"] = (int(height), int(kwargs["frame_width"]))
    else:
        kwargs["output_shape"] = (int(data_shape[0]), int(data_shape[1]))

    kwargs["output_shape"] = (int(kwargs["output_shape"][1]), int(kwargs["output_shape"][0]))
    writer_kwargs = extract_method_kwargs(kwargs, VideoWriter.__init__)
    print(writer_kwargs)
    with VideoWriter(str(file_name), **writer_kwargs) as writer:
        for i in range(nframes):
            writer.write(data[i])

    if show == "ipython":
        return IPython.display.Video(str(file_name), height=player_height, width=player_width)
    elif show == "opencv":
        raise NotImplementedError("Still working on this - use ipython for now")
    return None
