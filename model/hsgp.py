import jax.numpy as jnp 
import jax
from numpyro import sample
from numpyro.distributions import Distribution
import numpyro.distributions as dist


def spectral_density(w, alpha, length):
    c = alpha * jnp.sqrt(2 * jnp.pi) * length
    e = jnp.exp(-0.5 * (length**2) * (w**2))
    return c * e

def sqrt_eigenvalues(M, L):
    return jnp.arange(1,  M + 1) * jnp.pi / (2 * L) 

def convex_eigenfunctions(x, L, M= 1):
    assert len(x.shape) == 1 ### only have capacity for single dimension concavity
    eig_vals = sqrt_eigenvalues(M, L)
    broadcast_sub = jax.vmap(jax.vmap(jnp.subtract, (None, 0)), (0, None))
    broadcast_add = jax.vmap(jax.vmap(jnp.add, (None, 0)), (0, None))
    sum_eig_vals = broadcast_add(eig_vals, eig_vals)
    diff_eig_vals = broadcast_sub(eig_vals, eig_vals)
    diff_eig_vals_square = jnp.power( diff_eig_vals, 2)
    sum_eig_vals_square = jnp.power(sum_eig_vals, 2)
    x_shifted = x + L
    cos_pos = jnp.cos(jnp.einsum("t,m... -> tm...", x_shifted, sum_eig_vals)) / (2 * L * sum_eig_vals_square)
    cos_neg = jnp.cos(jnp.einsum("t,m... -> tm...", x_shifted, diff_eig_vals)) / (2 * L * diff_eig_vals_square)

    diagonal_constants = broadcast_sub(jnp.power(x, 2) / (4 * L) + (x / 2) + (L/ 4) , (1 / (2 * L * jnp.diagonal(sum_eig_vals_square)))) ### should be t x m
    diagonal_elements = diagonal_constants + jnp.diagonal(sum_eig_vals_square) / (2 * L)

    other_elements = cos_pos - cos_neg 
    broadcast_fill_diag = jax.vmap(lambda x, y: jnp.fill_diagonal(x,y, inplace=False), in_axes = 0)
    phi = broadcast_fill_diag(other_elements, diagonal_elements)
    return phi #should be t x m x m where t is the length of x and m is the number of eigen values (or M)


def diag_spectral_density(alpha, length, L, M):
    return spectral_density(sqrt_eigenvalues(M, L), alpha, length)


def eigenfunctions(x, L, M):
    """
    The first `M` eigenfunctions of the laplacian operator in `[-L, L]`
    evaluated at `x`. These are used for the approximation of the
    squared exponential kernel.
    """
    m1 = (jnp.pi / (2 * L)) * jnp.tile(L + x[:, None], M)
    m2 = jnp.diag(jnp.linspace(1, M, num=M))
    num = jnp.sin(m1 @ m2)
    den = jnp.sqrt(L)
    return num / den


def make_convex_f(phi, gamma, intercept, slope, x, L):
    right_result = jax.vmap(jax.vmap(lambda x: jnp.einsum("tjk,k -> tj", phi, x), -1, -1), -1, -1)(gamma)
    result = jax.vmap(jax.vmap(lambda x, y: jnp.einsum("tj,j -> t", x, y), -1, -1), -1, -1)(right_result, gamma)
     ## output is length of x and output size (mult by -1 to make sure we get concave functions)    
    return jnp.swapaxes(intercept + jnp.einsum("o,t -> to", slope, (x + L))[:, None, ...] - result, 0, -1)


def make_gamma(weights, alpha, length, M, L, output_size):
    spd = jnp.moveaxis(jnp.tile(jnp.sqrt(diag_spectral_density(alpha, length, L, M)), (*output_size,1)), -1, 0)
    gamma = spd * weights
    return gamma

# --- Approximate Gaussian processes --- #
def approx_se_ncp(x, alpha, length, L, M, output_size = 1, name: str = ""):
    """
    Hilbert space approximation for the squared
    exponential kernel in the non-centered parametrisation.
    """
    phi = eigenfunctions(x, L, M)
    spd = jnp.tile(jnp.sqrt(diag_spectral_density(alpha, length, L, M)), (output_size,1)).T
    beta = sample(name, dist.Normal(0, 1), sample_shape=(M, output_size)) ### can have multi-output
    f = phi @ (spd * beta)
    return f


def approx_convex_se(phi, x, alpha, length, L, M, output_size, beta, intercept, slope, name: str = ""):
    """
    Hilbert space approximation for the squared
    exponential kernel in the non-centered parametrisation for convex gp based on 
    https://drive.google.com/file/d/19kHhfuYC22ymNt3a3BlYgW0vWexOmbNy/view
    """

    weights = sample(name, beta, sample_shape=(M, *output_size)) if isinstance(beta, Distribution) else beta ### can have multi-output
    gamma = make_gamma(weights, alpha, length, M, L, output_size)
    f_0 = sample("f_0", intercept, sample_shape = (1, output_size[-1])) if isinstance(intercept, Distribution) else intercept
    f_0_prime = sample("f_0_prime", slope, sample_shape = (output_size[-1], )) if isinstance(slope, Distribution) else slope
    return make_convex_f(phi, gamma, f_0, f_0_prime, x, L)