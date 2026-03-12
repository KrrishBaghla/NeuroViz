from engine.nn import MLP
from engine.value import Value

model = MLP(2, [4, 1])

x = [Value(1), Value(0)]
print(model(x))
