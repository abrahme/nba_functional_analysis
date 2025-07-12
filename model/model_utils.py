import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import constraints
from jax import random

class Type2Gumbel(dist.Distribution):
    support = constraints.positive
    arg_constraints = {'alpha': constraints.positive, 'scale': constraints.positive}
    reparametrized_params = ['alpha', 'scale']
    
    def __init__(self, alpha, scale=1.0, validate_args=None):
        self.alpha = alpha
        self.scale = scale
        batch_shape = jnp.shape(jnp.broadcast_arrays(alpha, scale)[0])
        super().__init__(batch_shape=batch_shape, event_shape=(), validate_args=validate_args)
    
    def sample(self, key, sample_shape=()):
        # Inverse CDF method:
        u = random.uniform(key, shape=sample_shape + self.batch_shape)
        return self.scale / (-jnp.log(u))**(1.0 / self.alpha)
    
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        alpha = self.alpha
        s = self.scale
        x = value
        log_unnormalized = jnp.log(alpha) + alpha * jnp.log(s) - (alpha + 1) * jnp.log(x)
        log_normalizer = (s / x) ** alpha
        return log_unnormalized - log_normalizer
