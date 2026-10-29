class Value:
	def __init__(self, data, _children=(), _op=''):
		self.data = data
		self.grad = 0
		self._backward = lambda: None
		self._prev = set(_children)
		self._op = _op
	def __repr__(self):
		def fmt(x):
			return int(x) if float(x).is_integer() else round(x, 4)
		return f"Value(data={fmt(self.data)}, grad={fmt(self.grad)})"

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
			self.grad += out.grad * other.data
			other.grad += out.grad * self.data
		
		out._backward = _backward

		return out

	def relu(self):
		out = Value(max(0, self.data), (self,), 'relu')

		def _backward():
			self.grad += out.grad * (self.data > 0)
		
		out._backward = _backward

		return out


	def backward(self):
		topo = []
		visited = set()
		
		def build_topo(u):
			if u not in visited:
				visited.add(u)
				for v in u._prev:
					build_topo(v)
				topo.append(u)
		
		build_topo(self)

		self.grad = 1

		for u in reversed(topo):
			u._backward()

