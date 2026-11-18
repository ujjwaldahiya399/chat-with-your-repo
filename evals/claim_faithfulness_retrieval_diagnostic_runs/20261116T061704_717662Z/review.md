# Micrograd claim faithfulness review

Expected claims and evidence below were never included in generation prompts.
Split each generated answer into small factual claims. Label each claim SUPPORTED, CONTRADICTED, or NOT_IN_SOURCE.

## H02

How does Neuron compute an activation when its input list contains ordinary Python numbers, and which Value operations keep the weighted sum differentiable?

Answerable: True

### Generated answer

The `Neuron` class treats the inputs as ordinary Python scalars, but each weight `wi` is a `Value`.  
When the neuron is called, it builds the activation with  

```python
act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
```  

For every pair `wi, xi` the expression `wi * xi` invokes **`Value.__mul__`** (because `wi` is a `Value` and `xi` is a plain number, which is first wrapped in a `Value`). This returns a new `Value` whose `_backward` closure records the derivative of the product ( `self.grad += other.data * out.grad`, `other.grad += self.data * out.grad` )【Source 3】.  

The `sum` starts from the bias `self.b` (also a `Value`) and adds each product using **`Value.__add__`**. Each addition creates another `Value` with a `_backward` that propagates the upstream gradient to both operands (`self.grad += out.grad`, `other.grad += out.grad`)【Source 3】.  

Thus the weighted sum remains a `Value` object that participates in the autograd graph, and the combination of `Value.__mul__` and `Value.__add__` (and, if `nonlin` is true, the subsequent `relu` which is also a `Value` operation) keeps the computation fully differentiable【Source 1】.

### Expected claims

- Neuron pairs weights and inputs with zip.
- Each paired weight and input are multiplied.
- Value.__mul__ converts a non-Value operand into a Value.
- Python sum starts the accumulated activation from the neuron's bias.
- Value.__add__ creates graph nodes and its backward closure accumulates gradients into both operands.
- The activation is passed through ReLU only when the neuron's nonlin flag is true.

### Claim annotations

| Claim | Exact claim text | Label | Cited source | Evidence span | Rationale |
|---|---|---|---|---|---|
| C1 |  |  |  |  |  |

### Retrieved sources

Source 1: micrograd/nn.py:13-29

````text
class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


````

Source 2: micrograd/nn.py:30-44

````text
class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


````

Source 3: micrograd/engine.py:2-94

````text
class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

````

Source 4: README.md:1-50

````text

# micrograd

![awww](puppy.jpg)

A tiny Autograd engine (with a bite! :)). Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are tiny, with about 100 and 50 lines of code respectively. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification, as the demo notebook shows. Potentially useful for educational purposes.

### Installation

```bash
pip install micrograd
```

### Example usage

Below is a slightly contrived example showing a number of possible supported operations:

```python
from micrograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
```

### Training a neural net

The notebook `demo.ipynb` provides a full demo of training an 2-layer neural network (MLP) binary classifier. This is achieved by initializing a neural net from `micrograd.nn` module, implementing a simple svm "max-margin" binary classification loss and using SGD for optimization. As shown in the notebook, using a 2-layer neural net with two 16-node hidden layers we achieve the following decision boundary on the moon dataset:

![2d neuron](moon_mlp.png)

### Training a GPT

For a more advanced example, see [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95), which trains and samples from a full GPT-2-like transformer in pure, dependency-free Python. It builds on a more efficient and better version of the autograd engine here (storing local gradients at forward time instead of per-op backward closures), and is the complete algorithm in a single file — everything else is just efficiency. See also the accompanying [explainer post](https://karpathy.github.io/2026/02/12/microgpt/) for a detailed walkthrough.

### Tracing / visualization


````

Source 5: micrograd/nn.py:45-60

````text
class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

````

## H05

When zero_grad() is called on an MLP, how does it reach every weight and bias, and what value is assigned to each gradient?

Answerable: True

### Generated answer

`zero_grad()` is defined in the base `Module` class to iterate over the list returned by `self.parameters()` and set each parameter’s `grad` field to 0 (Source 1).  

An `MLP` inherits from `Module` and overrides `parameters()` so that it returns a flat list of all parameters of its constituent layers (Source 3). Each `Layer` in turn returns the parameters of all its neurons (Source 5), and each `Neuron` returns its weight list `self.w` plus its bias `self.b` (Source 4).  

Thus, when `zero_grad()` is called on an `MLP`, the call walks through:

* MLP → its `layers` → each `Layer` → its `neurons` → each `Neuron` → its weights and bias,

collecting every weight and bias into the list processed by `zero_grad()`. For each of those `Value` objects the gradient attribute `grad` is set to 0.

### Expected claims

- MLP inherits zero_grad from Module.
- Module.zero_grad calls self.parameters().
- MLP.parameters flattens parameters from every layer.
- Layer.parameters flattens parameters from every neuron.
- Neuron.parameters returns its weights followed by its bias.
- zero_grad assigns 0 to every returned parameter's grad field.

### Claim annotations

| Claim | Exact claim text | Label | Cited source | Evidence span | Rationale |
|---|---|---|---|---|---|
| C1 |  |  |  |  |  |

### Retrieved sources

Source 1: micrograd/nn.py:4-12

````text
class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


````

Source 2: micrograd/engine.py:2-94

````text
class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

````

Source 3: micrograd/nn.py:45-60

````text
class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

````

Source 4: micrograd/nn.py:13-29

````text
class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


````

Source 5: micrograd/nn.py:30-44

````text
class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


````
