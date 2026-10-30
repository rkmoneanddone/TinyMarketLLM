"""Dhan adapter boundary. Live trading is deliberately unavailable in V1."""
class DhanAdapter:
    def __init__(self, access_token=None):
        self.access_token = access_token

    def connect(self):
        raise NotImplementedError("Connect after Dhan access is renewed and API fields verified")

    def place_order(self, *args, **kwargs):
        raise RuntimeError("LIVE ORDERS DISABLED: adapter has no execution implementation")
