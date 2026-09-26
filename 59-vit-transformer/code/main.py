"""Vision transformer encoder implemented from scratch with NumPy.

The model follows the ViT-Base/16 recipe: split a 224x224 RGB image into
16x16 patches, project each patch to 768 dimensions, prepend a learned CLS
token, and add a fixed 2D sinusoidal position signal. Twelve pre-LayerNorm
transformer blocks with twelve attention heads then contextualize the token
sequence. A final LayerNorm produces the contextual tokens, and the CLS row is
the pooled image feature.

The implementation is forward-first and float32-only. A manual backward pass is
included so the gradient-flow tests can prove that the pooled CLS depends on
the patch projection; it is not a general autograd engine or a trainer.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "Module",
    "Conv2d",
    "PatchEmbed",
    "VisionFrontEnd",
    "Linear",
    "LayerNorm",
    "gelu",
    "gelu_grad",
    "sinusoidal_2d",
    "synthesize_image",
]


class Module:
    """Minimal callable module with a parameter/gradient surface."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(f"{type(self).__name__} does not implement forward()")

    def parameters(self) -> list[np.ndarray]:
        return []

    def num_parameters(self) -> int:
        return sum(int(parameter.size) for parameter in self.parameters())

    def zero_grad(self) -> None:
        return None


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
        self.weight_grad = np.zeros_like(self.weight)
        self.bias_grad = np.zeros_like(self.bias)
        self._columns: np.ndarray | None = None
        self._input_shape: tuple[int, int, int, int] | None = None
        self._out_shape: tuple[int, int, int, int] | None = None

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

        self._columns = columns
        self._input_shape = x.shape
        self._out_shape = (batch, self.out_channels, out_h, out_w)
        return projected.reshape(batch, out_h, out_w, self.out_channels).transpose(
            0, 3, 1, 2
        )

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._columns is None or self._input_shape is None or self._out_shape is None:
            raise RuntimeError("forward() must run before backward()")
        dout = np.asarray(dout, dtype=np.float32)
        if dout.shape != self._out_shape:
            raise ValueError(
                f"expected gradient shape {self._out_shape}, got {dout.shape}"
            )

        batch, _, out_h, out_w = self._out_shape
        kernel = self.kernel_size
        flat = dout.transpose(0, 2, 3, 1).reshape(
            batch * out_h * out_w, self.out_channels
        )
        columns = self._columns.reshape(batch * out_h * out_w, -1)
        self.weight_grad += flat.T @ columns
        self.bias_grad += flat.sum(axis=0)

        dcolumns = (flat @ self.weight.reshape(self.out_channels, -1)).reshape(
            batch, out_h, out_w, self.in_channels, kernel, kernel
        )
        dcolumns = dcolumns.transpose(0, 3, 1, 2, 4, 5)
        dx = np.zeros(self._input_shape, dtype=np.float32)
        for kh in range(kernel):
            for kw in range(kernel):
                dx[:, :, kh : kh + out_h * self.stride : self.stride,
                   kw : kw + out_w * self.stride : self.stride] += dcolumns[
                    :, :, :, :, kh, kw
                ]
        return dx

    def parameters(self) -> list[np.ndarray]:
        return [self.weight, self.bias]

    def zero_grad(self) -> None:
        self.weight_grad.fill(0.0)
        self.bias_grad.fill(0.0)


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
        self._grid: tuple[int, int] | None = None

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
        self._grid = (grid_h, grid_w)
        return projected.transpose(0, 2, 3, 1).reshape(
            batch, grid_h * grid_w, self.hidden
        )

    def backward(self, dtokens: np.ndarray) -> np.ndarray:
        if self._grid is None:
            raise RuntimeError("forward() must run before backward()")
        grid_h, grid_w = self._grid
        if dtokens.ndim != 3 or dtokens.shape[1:] != (grid_h * grid_w, self.hidden):
            raise ValueError(
                f"expected gradient shape (batch, {grid_h * grid_w}, {self.hidden}), "
                f"got {dtokens.shape}"
            )
        dpatches = dtokens.reshape(
            dtokens.shape[0], grid_h, grid_w, self.hidden
        ).transpose(0, 3, 1, 2)
        return self.proj.backward(dpatches)

    def parameters(self) -> list[np.ndarray]:
        return self.proj.parameters()

    def zero_grad(self) -> None:
        self.proj.zero_grad()


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
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")

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
        self.cls_token_grad = np.zeros_like(self.cls_token)
        self._token_shape: tuple[int, int, int] | None = None

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
        tokens = np.concatenate((cls, patch_tokens), axis=1)
        self._token_shape = (batch, 1 + grid_h * grid_w, self.hidden)
        return tokens

    def backward(self, dtokens: np.ndarray) -> np.ndarray:
        if self._token_shape is None:
            raise RuntimeError("forward() must run before backward()")
        if dtokens.shape != self._token_shape:
            raise ValueError(
                f"expected gradient shape {self._token_shape}, got {dtokens.shape}"
            )
        dpatches = dtokens[:, 1:, :]
        self.cls_token_grad += dtokens[:, 0, :].sum(axis=0).reshape(1, 1, self.hidden)
        return self.patch_embed.backward(dpatches)

    def parameters(self) -> list[np.ndarray]:
        return self.patch_embed.parameters() + [self.cls_token]

    def zero_grad(self) -> None:
        self.patch_embed.zero_grad()
        self.cls_token_grad.fill(0.0)


def gelu(x: np.ndarray) -> np.ndarray:
    """Tanh approximation of GELU, differentiable without SciPy."""
    coefficient = np.sqrt(2.0 / np.pi)
    inner = coefficient * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def gelu_grad(x: np.ndarray) -> np.ndarray:
    """Derivative of :func:`gelu` with respect to its input."""
    coefficient = np.sqrt(2.0 / np.pi)
    inner = coefficient * (x + 0.044715 * x**3)
    tanh_inner = np.tanh(inner)
    return 0.5 * (1.0 + tanh_inner) + 0.5 * x * (1.0 - tanh_inner**2) * coefficient * (
        1.0 + 3.0 * 0.044715 * x**2
    )


class Linear(Module):
    """Affine projection ``y = x @ weight.T + bias`` with manual gradients."""

    def __init__(self, in_features: int, out_features: int, seed: int = 0) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.in_features = in_features
        self.out_features = out_features
        rng = np.random.default_rng(seed)
        self.weight = rng.normal(
            0.0, 0.02, size=(out_features, in_features)
        ).astype(np.float32)
        self.bias = np.zeros(out_features, dtype=np.float32)
        self.weight_grad = np.zeros_like(self.weight)
        self.bias_grad = np.zeros_like(self.bias)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected last dimension {self.in_features}, got {x.shape}"
            )
        self._x = x
        return x @ self.weight.T + self.bias

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("forward() must run before backward()")
        dy = np.asarray(dy, dtype=np.float32)
        if dy.shape != (*self._x.shape[:-1], self.out_features):
            raise ValueError(f"expected gradient shape {self._x.shape[:-1] + (self.out_features,)}, got {dy.shape}")
        x_flat = self._x.reshape(-1, self.in_features)
        dy_flat = dy.reshape(-1, self.out_features)
        self.weight_grad += dy_flat.T @ x_flat
        self.bias_grad += dy_flat.sum(axis=0)
        return (dy @ self.weight).reshape(self._x.shape)

    def parameters(self) -> list[np.ndarray]:
        return [self.weight, self.bias]

    def zero_grad(self) -> None:
        self.weight_grad.fill(0.0)
        self.bias_grad.fill(0.0)


class LayerNorm(Module):
    """Layer normalization over the last axis with affine parameters."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.dim = dim
        self.eps = eps
        self.gamma = np.ones(dim, dtype=np.float32)
        self.beta = np.zeros(dim, dtype=np.float32)
        self.gamma_grad = np.zeros_like(self.gamma)
        self.beta_grad = np.zeros_like(self.beta)
        self._xhat: np.ndarray | None = None
        self._inv_std: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim < 1 or x.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {x.shape}")
        mean = x.mean(axis=-1, keepdims=True)
        variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(variance + self.eps)
        xhat = (x - mean) * inv_std
        self._xhat = xhat
        self._inv_std = inv_std
        return self.gamma * xhat + self.beta

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._xhat is None or self._inv_std is None:
            raise RuntimeError("forward() must run before backward()")
        dy = np.asarray(dy, dtype=np.float32)
        if dy.shape != self._xhat.shape:
            raise ValueError(
                f"expected gradient shape {self._xhat.shape}, got {dy.shape}"
            )
        axes = tuple(range(dy.ndim - 1))
        self.gamma_grad += (dy * self._xhat).sum(axis=axes)
        self.beta_grad += dy.sum(axis=axes)
        dxhat = dy * self.gamma
        dx = (self._inv_std / self.dim) * (
            self.dim * dxhat
            - dxhat.sum(axis=-1, keepdims=True)
            - self._xhat * (dxhat * self._xhat).sum(axis=-1, keepdims=True)
        )
        return dx

    def parameters(self) -> list[np.ndarray]:
        return [self.gamma, self.beta]

    def zero_grad(self) -> None:
        self.gamma_grad.fill(0.0)
        self.beta_grad.fill(0.0)


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
