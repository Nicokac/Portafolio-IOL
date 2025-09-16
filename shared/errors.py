"""Custom exception hierarchy shared across the application."""


class NetworkError(Exception):
    """Se lanza ante problemas de conectividad con servicios externos."""


class ExternalAPIError(Exception):
    """Se lanza cuando un proveedor externo responde con error o está inaccesible."""


__all__ = ["NetworkError", "ExternalAPIError"]
