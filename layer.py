class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def fwd_prop(self, input):
        raise NotImplementedError

    def bwd_prop(self, input):
        raise NotImplementedError
