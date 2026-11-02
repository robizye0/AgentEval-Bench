"""Rollback-only state sandboxing for idempotent mock tool executions."""

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, MutableMapping, MutableSequence, MutableSet
from functools import wraps
from types import TracebackType
from typing import Any


class sandbox_state:
    """Snapshot ``state`` and restore it on every exit, including successful calls.

    Use as ``with sandbox_state(db):`` or ``@sandbox_state(db)``. Mutable mock
    containers and objects with ``__dict__`` are copied deeply. Adapter objects
    can provide ``sandbox_snapshot()`` and ``sandbox_restore(snapshot)`` methods
    for database/API-specific rollback without coupling the library to a driver.
    """

    def __init__(self, state: Any):
        self.state = state
        self._snapshot: Any = None
        self._restore: Callable[[Any], None] | None = None

    def _recreate_cm(self) -> sandbox_state:
        return type(self)(self.state)

    def __enter__(self) -> Any:
        state = self.state
        snapshot = getattr(state, "sandbox_snapshot", None)
        restore = getattr(state, "sandbox_restore", None)
        if callable(snapshot) and callable(restore):
            self._snapshot = snapshot()
            self._restore = restore
        elif isinstance(state, MutableMapping):
            self._snapshot = copy.deepcopy(dict(state))
            self._restore = lambda saved: (state.clear(), state.update(saved))
        elif isinstance(state, MutableSequence):
            self._snapshot = copy.deepcopy(list(state))
            self._restore = lambda saved: state.__setitem__(slice(None), saved)
        elif isinstance(state, MutableSet):
            self._snapshot = copy.deepcopy(set(state))
            self._restore = lambda saved: (state.clear(), state.update(saved))
        elif isinstance(state, bytearray):
            self._snapshot = bytes(state)
            self._restore = lambda saved: state.__setitem__(slice(None), saved)
        elif hasattr(state, "__dict__"):
            self._snapshot = copy.deepcopy(vars(state))

            def restore_attributes(saved: dict[str, Any]) -> None:
                vars(state).clear()
                vars(state).update(saved)

            self._restore = restore_attributes
        else:
            raise TypeError(
                "sandbox state must be mutable or implement sandbox_snapshot()/sandbox_restore()"
            )
        return state

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if self._restore is not None:
            self._restore(self._snapshot)
        self._restore = None
        self._snapshot = None
        return False

    def __call__(self, function: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(function):
            @wraps(function)
            async def async_wrapped(*args: Any, **kwargs: Any) -> Any:
                with self._recreate_cm():
                    return await function(*args, **kwargs)

            return async_wrapped

        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with self._recreate_cm():
                return function(*args, **kwargs)

        return wrapped
