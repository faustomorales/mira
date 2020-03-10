class Object():
    def __init__(self, *args, **kwargs):
        self.__name__ = ''
        self.__qualname__ = ''
        self.__annotations__ = {}

    def __getattr__(self, item):
        return Object()

    def __dir__(self):
        return []

    def __call__(self, *args, **kwargs):
        return Object()

    def __mro_entries__(self, _):
        return (Object(), )
