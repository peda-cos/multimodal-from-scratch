"""Patch tokenization front end for a vision transformer, in NumPy.

The image is split into non-overlapping square patches, each patch is
projected by a ``Conv2d`` whose kernel and stride both equal the patch size,
and a fixed 2D sinusoidal signal is added to encode each patch's grid
position. A learned CLS vector is prepended as token zero.

This is a forward-only reference implementation of the front end described by
the ViT paper. The ``Module`` base mirrors the callable/parameter surface of
``torch.nn.Module`` without pulling in autograd.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "Module",
    "Conv2d",
    "PatchEmbed",
    "VisionFrontEnd",
    "sinusoidal_2d",
    "synthesize_image",
    "demo",
]


class Module:
    """Minimal callable module with a parameters list."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, _x: np.ndarray) -> np.ndarray:
        raise TypeError(f"{type(self).__name__} does not implement forward()")

    def parameters(self) -> list[np.ndarray]:
        return []


class Conv2d(Module):
    """Valid 2D convolution with no padding, stored as NCHW.

    Weight layout is ``(out_channels, in_channels, kernel, kernel)``, matching
    the PyTorch convention. Setting ``stride`` equal to ``kernel_size`` makes
    non-overlapping patches, which is the ViT patch projection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int | None = None,
        seed: int = 0,
    ) -> None:
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if stride is not None and stride <= 0:
            raise ValueError("stride must be positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride

        rng = np.random.default_rng(seed)
        self.weight = rng.normal(
            0.0,
            0.02,
            size=(out_channels, in_channels, kernel_size, kernel_size),
        ).astype(np.float32)
        self.bias = np.zeros(out_channels, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4:
            raise ValueError(f"expected a 4-D NCHW input, got shape {x.shape}")
        batch, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {channels}"
            )
        kernel = self.kernel_size
        if height < kernel or width < kernel:
            raise ValueError(
                f"input {height}x{width} is smaller than kernel {kernel}x{kernel}"
            )

        out_h = (height - kernel) // self.stride + 1
        out_w = (width - kernel) // self.stride + 1
        windows = sliding_window_view(x, (kernel, kernel), axis=(2, 3))
        windows = windows[:, :, :: self.stride, :: self.stride]
        columns = windows.transpose(0, 2, 3, 1, 4, 5).reshape(
            batch, out_h * out_w, channels * kernel * kernel
        )
        projected = columns @ self.weight.reshape(self.out_channels, -1).T
        projected += self.bias
        return projected.reshape(
            batch, out_h, out_w, self.out_channels
        ).transpose(0, 3, 1, 2)

    def parameters(self) -> list[np.ndarray]:
        return [self.weight, self.bias]


class PatchEmbed(Module):
    """Project an NCHW image into ``(batch, num_patches, hidden)`` tokens."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden: int = 768,
        patch_size: int = 16,
        seed: int = 0,
    ) -> None:
        if in_channels <= 0 or hidden <= 0:
            raise ValueError("in_channels and hidden must be positive")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")

        self.in_channels = in_channels
        self.hidden = hidden
        self.patch_size = patch_size
        self.proj = Conv2d(
            in_channels,
            hidden,
            kernel_size=patch_size,
            stride=patch_size,
            seed=seed,
        )

    def grid_shape(self, height: int, width: int) -> tuple[int, int]:
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        if height % self.patch_size != 0:
            raise ValueError(
                f"input height {height} is not divisible by patch size {self.patch_size}"
            )
        if width % self.patch_size != 0:
            raise ValueError(
                f"input width {width} is not divisible by patch size {self.patch_size}"
            )
        return height // self.patch_size, width // self.patch_size

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 4:
            raise ValueError(f"expected a 4-D NCHW input, got shape {x.shape}")
        batch, _, height, width = x.shape
        grid_h, grid_w = self.grid_shape(height, width)
        projected = self.proj(x)
        return projected.transpose(0, 2, 3, 1).reshape(
            batch, grid_h * grid_w, self.hidden
        )

    def parameters(self) -> list[np.ndarray]:
        return self.proj.parameters()


def _sinusoidal_1d(positions: np.ndarray, dim: int, base: float) -> np.ndarray:
    """Standard 1D sin/cos table with a sin/cos pair per frequency."""
    if dim <= 0 or dim % 2 != 0:
        raise ValueError("dim must be a positive even number")
    frequencies = np.exp(-np.log(base) * np.arange(0, dim, 2) / dim)
    angles = positions.astype(np.float64)[:, None] * frequencies[None, :]
    table = np.empty((positions.shape[0], dim), dtype=np.float64)
    table[:, 0::2] = np.sin(angles)
    table[:, 1::2] = np.cos(angles)
    return table


def sinusoidal_2d(
    grid_h: int,
    grid_w: int,
    dim: int,
    base: float = 10_000.0,
) -> np.ndarray:
    """Build a fixed row-major 2D sinusoidal table of shape ``(h*w, dim)``.

    The first ``dim // 2`` channels encode the row with a 1D sin/cos table; the
    remaining channels encode the column. Each frequency contributes one
    adjacent sin/cos pair, so every token has the same L2 norm,
    ``sqrt(dim / 2)``.
    """
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("grid_h and grid_w must be positive")
    if dim <= 0 or dim % 4 != 0:
        raise ValueError("dim must be a positive multiple of 4")
    if base <= 0.0:
        raise ValueError("base must be positive")

    row_dim = dim // 2
    col_dim = dim - row_dim
    rows = _sinusoidal_1d(np.arange(grid_h), row_dim, base)
    cols = _sinusoidal_1d(np.arange(grid_w), col_dim, base)
    row_part = np.repeat(rows, grid_w, axis=0)
    col_part = np.tile(cols, (grid_h, 1))
    return np.concatenate((row_part, col_part), axis=1).astype(np.float32)


class VisionFrontEnd(Module):
    """Patch projection, 2D position addition, and CLS prepend."""

    def __init__(
        self,
        patch_size: int = 16,
        hidden: int = 768,
        in_channels: int = 3,
        seed: int = 0,
    ) -> None:
        if hidden <= 0 or hidden % 4 != 0:
            raise ValueError("hidden must be a positive multiple of 4")

        self.patch_size = patch_size
        self.hidden = hidden
        self.in_channels = in_channels
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            hidden=hidden,
            patch_size=patch_size,
            seed=seed,
        )
        cls_rng = np.random.default_rng(seed + 1)
        self.cls_token = cls_rng.normal(
            0.0, 0.02, size=(1, 1, hidden)
        ).astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4:
            raise ValueError(f"expected a 4-D NCHW input, got shape {x.shape}")
        batch, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {channels}"
            )

        grid_h, grid_w = self.patch_embed.grid_shape(height, width)
        patch_tokens = self.patch_embed(x)
        positions = sinusoidal_2d(grid_h, grid_w, self.hidden)
        patch_tokens = patch_tokens + positions[None, :, :]
        cls = np.broadcast_to(self.cls_token, (batch, 1, self.hidden))
        return np.concatenate((cls, patch_tokens), axis=1)

    def parameters(self) -> list[np.ndarray]:
        return self.patch_embed.parameters() + [self.cls_token]


def synthesize_image(
    seed: int = 0,
    size: int = 224,
    in_channels: int = 3,
) -> np.ndarray:
    """Return a deterministic ``(1, C, size, size)`` float32 fixture in [0, 1)."""
    if size <= 0 or in_channels <= 0:
        raise ValueError("size and in_channels must be positive")
    rng = np.random.default_rng(seed)
    image = rng.random((size, size, in_channels), dtype=np.float32)
    return image.transpose(2, 0, 1)[None, :, :, :]


def demo() -> None:
    """Run one fixture through the default ViT-Base/16 front end."""
    image = synthesize_image(seed=7)
    model = VisionFrontEnd(seed=0)
    tokens = model(image)

    grid_h, grid_w = model.patch_embed.grid_shape(*image.shape[2:])
    positions = sinusoidal_2d(grid_h, grid_w, model.hidden)
    norms = np.linalg.norm(positions, axis=1)

    model(image)  # warm-up
    timings: list[float] = []
    for _ in range(7):
        started = perf_counter()
        model(image)
        timings.append(perf_counter() - started)
    median_ms = 1_000.0 * float(np.median(timings))

    print(f"image shape: {image.shape}")
    print(f"patch grid: {grid_h} x {grid_w} = {grid_h * grid_w}")
    print(f"token shape: {tokens.shape}")
    print(f"CLS norm: {float(np.linalg.norm(tokens[0, 0])):.6f}")
    print(
        "position row 0[:8]: "
        + np.array2string(positions[0, :8], precision=6, floatmode="fixed")
    )
    print(
        "position token norms: "
        f"min={float(norms.min()):.6f} max={float(norms.max()):.6f}"
    )
    print(f"forward median (7 runs, batch=1): {median_ms:.3f} ms")


if __name__ == "__main__":
    demo()
