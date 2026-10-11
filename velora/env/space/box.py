# Copyright 2026 Achronus
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from typing import Self

import torch

from velora.env.space.base import RLSpace


def _tensor_short_repr(x: torch.Tensor) -> str:
    """
    Create a shorted string representation of a PyTorch tensor.

    Parameters
    ----------
    x : torch.Tensor
        The tensor to represent

    Returns
    -------
    repr : str
        A short representation of the tensor
    """
    if x.size != 0 and torch.min(x) == torch.max(x):
        return str(torch.min(x).item())

    return str(x)


class _Sampler:
    """
    Storage container for precomputed sampling data for box space.

    All masks and safe bounds are computed once, so sampling and rescaling
    contain no data-dependent branching (no host syncs, CUDA graph safe).

    Parameters
    ----------
    low : torch.Tensor
        Lower bound(s). Use `-inf` for unbounded lower
    high : torch.Tensor
        Upper bound(s). Use `-inf` for unbounded higher
    """

    def __init__(self, low: torch.Tensor, high: torch.Tensor) -> None:
        below = torch.isfinite(low)
        above = torch.isfinite(high)

        self._both = below & above

        self._low_only = below & ~above
        self._high_only = ~below & above

        # Infinite bound handling
        self._safe_low = torch.where(below, low, 0.0)
        self._safe_high = torch.where(above, high, 0.0)

        # Full bounded handling
        self._is_bounded = bool(self._both.all())
        self._mid = (self._safe_low + self._safe_high) / 2
        self._half = (self._safe_high - self._safe_low) / 2
        self._half_safe = torch.where(
            self._half > 0,
            self._half,
            torch.ones_like(self._half),
        )

    @property
    def is_bounded(self) -> bool:
        """Whether every dimension has both a finite lower and upper bound."""
        return self._is_bounded

    def bounded(self, unit: torch.Tensor) -> torch.Tensor:
        """
        Compute uniformly distributed values within the bounds.

        Dimensions without two finite bounds hold meaningless values here
        and are masked out by `sample()`.

        Parameters
        ----------
        unit : torch.Tensor
            Uniform samples on `[0, 1)` `(*shape)` or `(batch_size, *shape)`

        Returns
        -------
        bounded : torch.Tensor
            Values on `[low, high]`, same shape as `unit`
        """
        return self._safe_low + (self._safe_high - self._safe_low) * unit

    def sample(self, unit: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
        """
        Combine random draws into samples according to each dimension's bounds.

        - both bounds finite: uniform, from `unit`
        - only lower bound finite: lower bound plus an exponential
        - only upper bound finite: upper bound minus an exponential
        - neither bound finite: standard normal, from `normal`

        Parameters
        ----------
        unit : torch.Tensor
            Uniform samples on `[0, 1)` `(*shape)` or `(batch_size, *shape)`
        normal : torch.Tensor
            Standard normal samples, same shape as `unit`

        Returns
        -------
        samples : torch.Tensor
            Samples respecting the bounds, same shape as `unit`
        """
        expo = -torch.log1p(-unit)  # Exp(1) derived from the same uniform draw

        out = torch.where(self._both, self.bounded(unit), normal)
        out = torch.where(self._low_only, self._safe_low + expo, out)
        return torch.where(self._high_only, self._safe_high - expo, out)

    def rescale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map values from `[low, high]` to `[-1, 1]`.

        Dimensions where `low == high` map to `0`. Only meaningful when
        `is_bounded` is `True`.

        Parameters
        ----------
        x : torch.Tensor
            Values in the box `(*shape)` or `(batch_size, *shape)`

        Returns
        -------
        scaled : torch.Tensor
            Values in `[-1, 1]`, same shape as `x`
        """
        return (x - self._mid) / self._half_safe

    def unscale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map values from `[-1, 1]` back to `[low, high]`, the inverse of `rescale()`.

        Only meaningful when `is_bounded` is `True`.

        Parameters
        ----------
        x : torch.Tensor
            Values in `[-1, 1]` `(*shape)` or `(batch_size, *shape)`

        Returns
        -------
        unscaled : torch.Tensor
            Values in `[low, high]`, same shape as `x`
        """
        return self._mid + x * self._half


class BoxSpace(RLSpace):
    """
    A (possibly unbounded) box in `R^n`, with per-dimension bounds.

    Bounds are stored as device tensors expanded to the full `shape`.
    Sampling follows the Gymnasium convention per dimension:

    - both bounds finite: uniform on `[low, high]`
    - one bound finite: shifted exponential away from the finite bound
    - neither bound finite: standard normal

    Parameters
    ----------
    low : float | torch.Tensor
        Lower bound(s). Use `-inf` for unbounded lower
    high : float | torch.Tensor
        Upper bound(s). Use `-inf` for unbounded higher
    shape : tuple[int, ...] (optional)
        Shape of the space. When `None`, inferred by
        broadcasting `low` and `high`. Default is `None`
    dtype : torch.dtype (optional)
        FLoating point dtype. Default is `torch.float32`
    device : torch.device | str (optional)
        Device the bounds and samples live on. Default is `"cuda"`

    Raises
    ------
    incorrect_dtype : ValueError
        Errors when 'dtype' is not a floating point variant
    low_too_high : ValueError
        Errors when 'low' is <= 'high'
    """

    def __init__(
        self,
        low: float | torch.Tensor,
        high: float | torch.Tensor,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cuda",
    ) -> None:
        if not dtype.is_floating_point:
            raise ValueError(f"'dtype' must be a floating dtype. Got: '{dtype}'")

        low_t = torch.as_tensor(low, dtype=dtype, device=device)
        high_t = torch.as_tensor(high, dtype=dtype, device=device)
        shape = shape or tuple(torch.broadcast_shapes(low_t.shape, high_t.shape))

        super().__init__(shape, dtype, device=device)

        self._low = low_t.expand(shape).clone()
        self._high = high_t.expand(shape).clone()

        if (self._low > self._high).any():
            raise ValueError("'low' must be <= 'high'")

        self.low_repr: str | None = None
        self.high_repr: str | None = None

        self._sampler = _Sampler(self._low, self._high)

    @property
    def low(self) -> torch.Tensor:
        """Lower bounds."""
        return self._low

    @property
    def high(self) -> torch.Tensor:
        """Upper bounds."""
        return self._high

    @property
    def is_bounded(self) -> bool:
        """Whether every dimension has a finite lower and upper bound."""
        return self._sampler.is_bounded

    @classmethod
    def symmetric(
        cls,
        limit: float | torch.Tensor,
        shape: tuple[int, ...] | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cuda",
    ) -> Self:
        """
        Create a box symmetric about zero, `[-limit, limit]`.

        Parameters
        ----------
        limit : float | torch.Tensor
            Magnitude of the bounds. Must be non-negative
        shape : tuple[int, ...] (optional)
            Shape of a single element. When `None`, inferred from `limit`.
            Default is `None`
        dtype : torch.dtype (optional)
            Floating point dtype. Default is `torch.float32`
        device : torch.device | str (optional)
            Device the bounds and samples live on. Default is `"cuda"`

        Returns
        -------
        space : BoxSpace
            The symmetric box
        """
        limit_t = torch.as_tensor(limit, dtype=dtype, device=device)
        return cls(
            -limit_t,
            limit_t,
            shape=shape,
            dtype=dtype,
            device=device,
        )

    def sample(
        self,
        batch_size: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        out_shape = self.shape if batch_size is None else (batch_size, *self.shape)

        unit = torch.rand(
            out_shape,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        normal = torch.randn(
            out_shape,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        return self._sampler.sample(unit, normal)

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        """
        Check whether elements lie within the bounds (inclusive).

        Parameters
        ----------
        x : torch.Tensor
            Element(s) to check `(*shape)` or `(batch_size, *shape)`

        Returns
        -------
        mask : torch.Tensor
            Membership result, `bool`. Shape `()` for a single element,
            otherwise `(batch_size,)`. Elements containing NaN are `False`

        Raises
        ------
        x_shape_mismatch : ValueError
            If the trailing dimensions of `x` don't match `shape`.
        """
        n = len(self.shape)

        if tuple(x.shape[x.ndim - n :]) != self.shape:
            raise ValueError(
                f"Expected trailing shape '{self.shape}'. Got: '{tuple(x.shape)}'"
            )

        ok = (x >= self._low) & (x <= self._high)  # NaN compares as False
        return ok.flatten(x.ndim - n).all(-1) if n else ok

    def clip(self, x: torch.Tensor) -> torch.Tensor:
        """
        Clamp values into the bounds.

        Parameters
        ----------
        x : torch.Tensor
            Values to clamp `(*shape)` or `(batch_size, *shape)`

        Returns
        -------
        clipped : torch.Tensor
            Clamped values, same shape as `x`
        """
        return torch.clamp(x, self._low, self._high)

    def rescale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map values from `[low, high]` to `[-1, 1]`.

        Dimensions where `low == high` map to `0`.

        Parameters
        ----------
        x : torch.Tensor
            Values in the box `(*shape)` or `(batch_size, *shape)`

        Returns
        -------
        scaled : torch.Tensor
            Values in `[-1, 1]`, same shape as `x`

        Raises
        ------
        not_bounded : ValueError
            If the space is not fully bounded.
        """
        if not self._sampler.is_bounded:
            raise ValueError("Rescaling requires every dimension to have finite bounds")

        return self._sampler.rescale(x)

    def unscale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map values from `[-1, 1]` back to `[low, high]`, the inverse of `rescale()`.

        Typically used to turn a normalised policy output into an action.

        Parameters
        ----------
        x : torch.Tensor
            Values in `[-1, 1]` `(*shape)` or `(batch_size, *shape)`

        Returns
        -------
        unscaled : torch.Tensor
            Values in `[low, high]`, same shape as `x`

        Raises
        ------
        not_bounded : ValueError
            If the space is not fully bounded.
        """
        if not self._sampler.is_bounded:
            raise ValueError("Unscaling requires every dimension to have finite bounds")

        return self._sampler.unscale(x)

    def to(self, device: torch.device | str) -> Self:
        """
        Create a copy of the space on another device.

        Parameters
        ----------
        device : torch.device | str
            Target device

        Returns
        -------
        space : BoxSpace
            New space with identical bounds, shape and dtype on `device`
        """
        return type(self)(
            self._low,
            self._high,
            shape=self.shape,
            dtype=self.dtype,
            device=device,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoxSpace):
            return False

        return (
            self.shape == other.shape
            and self.dtype == other.dtype
            and torch.equal(self._low, other._low.to(self.device))
            and torch.equal(self._high, other._high.to(self.device))
        )

    def __repr__(self) -> str:
        if self.low_repr is None:
            self.low_repr = _tensor_short_repr(self._low)
        if self.high_repr is None:
            self.high_repr = _tensor_short_repr(self._high)

        return f"{self.__class__.__name__}({self.low_repr}, {self.high_repr}, {self.shape}, {self.dtype})"
