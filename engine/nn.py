import random
from engine.value import Value

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)

    def __call__(self, x):
        out = self.b
        for wi, xi in zip(self.w, x):
            out = out + wi * xi
        return out.relu()

    def parameters(self):
        return self.w + [self.b]


class MLP:
    def __init__(self, nin, layers):
        sizes = [nin] + layers
        self.layers = []

        for i in range(len(layers)):
            self.layers.append(
                [Neuron(sizes[i]) for _ in range(sizes[i + 1])]
            )

    def __call__(self, x):
        for layer in self.layers:
            x = [n(x) for n in layer]
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for n in layer for p in n.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
