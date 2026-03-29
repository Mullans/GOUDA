"""Misc. metrics to evaluate images."""

import cv2
import numpy as np
import numpy.typing as npt

__all__ = ["laplacian_var", "sobel_var"]


def laplacian_var(image: npt.NDArray) -> float:
    """Return the laplacian variance of an image."""
    # Laplacian is the rate of change of pixel intensity (2nd order derivative)
    blur = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=0.0, sigmaY=0.0)
    grey = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    result: np.float64 = cv2.Laplacian(grey, cv2.CV_16S).var()
    return float(result)


def sobel_var(image: npt.NDArray) -> float:
    """Return the sobal variance of an image."""
    # Sobel is the gradient of pixel intensity (1st order derivative)
    blur = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=0.0, sigmaY=0.0)
    grey = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(grey, ddepth=cv2.CV_16S, dx=1, dy=0, ksize=3)
    grad_x = cv2.convertScaleAbs(grad_x)
    grad_y = cv2.Sobel(grey, ddepth=cv2.CV_16S, dx=0, dy=1, ksize=3)
    grad_y = cv2.convertScaleAbs(grad_y)
    result: np.float64 = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0).var()
    return float(result)
