"""Authentication fixtures for tests."""

class FakeAuth:
    """Reusable authentication stub for infrastructure and API tests."""

    def __init__(self, access="FAKE_TOKEN", refresh="REFRESH_TOKEN", expired=False):
        self.access_token = access
        self.refresh_token = refresh
        self.tokens = {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
        }
        self._expired = expired

    def is_expired(self):
        """Return whether the fake token is expired."""
        return self._expired

    def mark_expired(self):
        """Simulate token expiration."""
        self._expired = True
