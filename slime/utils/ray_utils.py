class Box:
    """Protect the passed object"""
    def __init__(self, inner):
        self._inner = inner

    @property
    def inner(self):
        return self._inner
