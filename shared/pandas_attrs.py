"""Helpers para manipular atributos de pandas sin romper copias profundas."""

from __future__ import annotations

from types import MethodType
from typing import Any, Callable
import weakref


class SerializableCallable:
    """Envoltura ligera que evita copiar objetos no serializables."""

    __slots__ = ("_weak_method", "_callable", "_repr")

    def __init__(self, func: Callable[..., Any]):
        self._repr = repr(func)
        if isinstance(func, MethodType):
            self._weak_method: weakref.WeakMethod | None = weakref.WeakMethod(func)
            self._callable: Callable[..., Any] | None = None
        else:
            self._weak_method = None
            self._callable = func

    def resolve(self) -> Callable[..., Any] | None:
        if self._weak_method is not None:
            return self._weak_method()
        return self._callable

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - passthrough
        func = self.resolve()
        if func is None:
            raise RuntimeError("market price fetcher no longer available")
        return func(*args, **kwargs)

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.resolve() is not None

    def __deepcopy__(self, memo: dict[int, object]) -> "SerializableCallable":
        return self

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return self._repr

    __str__ = __repr__


def wrap_callable_attr(value: Callable[..., Any] | SerializableCallable | None):
    """Devuelve una envoltura segura para métodos ligados a objetos."""

    if value is None or isinstance(value, SerializableCallable):
        return value
    if isinstance(value, MethodType):
        return SerializableCallable(value)
    return value


def unwrap_callable_attr(value: Callable[..., Any] | SerializableCallable | None):
    """Recupera el callable original almacenado en los attrs."""

    if isinstance(value, SerializableCallable):
        return value.resolve()
    return value

