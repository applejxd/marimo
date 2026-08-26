import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a href="https://colab.research.google.com/github/applejxd/colaboratory/blob/master/algorithm/AutomaticDerivative.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Automatic Derivative by myself
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [Automatic Derivatives - Ceres Solver](http://ceres-solver.org/automatic_derivatives.html)
    """)
    return


@app.cell
def _():
    import numpy as np

    class Jet:

        def __init__(self, a: float, v):
            self.a = a
            self.v = np.array(v)

        def __str__(self):
            return f'{self.a}+{self.v}'

        def __add__(self, other):
            if isinstance(other, (int, float)):
                other = Jet(other, np.zeros(len(self.v)))
            return Jet(self.a + other.a, self.v + other.v)

        def __sub__(self, other):
            if isinstance(other, (int, float)):
                other = Jet(other, np.zeros(len(self.v)))
            return Jet(self.a - other.a, self.v - other.v)

        def __mul__(self, other):
            if isinstance(other, (int, float)):
                other = Jet(other, np.zeros(len(self.v)))
            return Jet(self.a * other.a, self.a * other.v + self.v * other.a)

        def __truediv__(self, other):
            if isinstance(other, (int, float)):
                other = Jet(other, np.zeros(len(self.v)))
            return Jet(self.a / other.a, self.v / other.a - self.a * other.v / other.a ** 2)

        def __pow__(self, other):
            if isinstance(other, (int, float)):
                other = Jet(other, np.zeros(len(self.v)))
            return Jet(self.a ** other.a, other.a * self.a ** (other.a - 1) * self.v + self.a ** other.a * np.log(self.a) * other.v)

        def __radd__(self, other):
            return self.__add__(other)

        def __rsub__(self, other):
            return self.__sub__(other)

        def __rmul__(self, other):
            return self.__mul__(other)
    _x = Jet(1, (2, 3))
    _y = Jet(4, np.array((5, 6)))
    print(_x + _y)
    print(2 * _x)
    return Jet, np


@app.cell
def _(Jet, np):
    def exp(x):
        return Jet(np.exp(x.a), np.exp(x.a) * x.v)
    print(exp(Jet(2, (3, 4))))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    \begin{equation}
    (\nabla (x^2+2y^2+3xy))(x=2, y=1)
    \end{equation}
    """)
    return


@app.cell
def _(Jet):
    def target_func(x, y):
        return x ** 2 + 2 * y ** 2 + 3 * x * y
    _x = Jet(2, (1, 0))
    # x=2
    _y = Jet(1, (0, 1))
    # y=1
    print(target_func(_x, _y))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Automatic Derivative by JAX
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From [the quickstart](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)
    """)
    return


@app.cell
def _():
    import jax.numpy as jnp
    from jax import grad

    return grad, jnp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    \begin{equation}
        \left.\sum_i (1+e^{-x_i})^{-1}\right|_{x_0=0, x_1=1, x_2=2}
    \end{equation}
    """)
    return


@app.cell
def _(grad, jnp):
    def sum_logistic(x):
        return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))
    x_small = jnp.arange(3.0)
    derivative_fn = grad(sum_logistic)
    print(derivative_fn(x_small))
    return sum_logistic, x_small


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    check numerical derivative solution
    """)
    return


@app.cell
def _(jnp, sum_logistic, x_small):
    def first_finite_differences(f, x):
        eps = 0.001
        return jnp.array([(f(x + eps * v) - f(x - eps * v)) / (2 * eps) for v in jnp.eye(len(x))])  # v_i is the tiny shift vector w.r.t. i-th component
    print(first_finite_differences(sum_logistic, x_small))
    return


if __name__ == "__main__":
    app.run()
