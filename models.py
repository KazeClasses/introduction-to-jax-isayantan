import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax as dx
from jaxtyping import Float, Array, Int, PRNGKeyArray

class CNNEmulator(eqx.Module):
    layers: list

    def __init__(self, key: PRNGKeyArray, hidden_dim: Int = 4):
        self.layers = [
            eqx.nn.Conv2d(2, 3, kernel_size=4, key=key1),
            jax.nn.tanh,
            eqx.nn.Conv2d(1, 3, kernel_size=4, key=key1)
        ]
        raise NotImplementedError

    def __call__(self, x: Float[Array, "2 n_res n_res"]) -> Float[Array, "1 n_res n_res"]:
        for layer in self.layers:
            x = layer(x)
        return x