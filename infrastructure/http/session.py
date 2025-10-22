# infrastructure/http/session.py
from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _wrap_with_timeout(request_func, default_timeout: float):
    def wrapped(method, url, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = default_timeout
        return request_func(method, url, **kwargs)

    return wrapped


def build_session(
    user_agent: str,
    *,
    retries: int = 2,
    backoff: float = 0.3,
    timeout: float = 15.0,
) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent})

    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    # envolver para tener timeout por defecto
    s.request = _wrap_with_timeout(s.request, timeout)
    return s
