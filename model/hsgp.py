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
    assert len(x.shape) == 1
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
    diagonal_constants = broadcast_sub(jnp.power(x, 2) / (4 * L) - (x / 2) + L / 4 , (1 / (2 * L * jnp.diag(diff_eig_vals_square)))) ### should be t x m

    diagonal_elements = diagonal_constants + jnp.diag(sum_eig_vals_square)

    other_elements = cos_pos - cos_neg 

    phi = jnp.fill_diagonal(other_elements, diagonal_elements, inplace=False)

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


def approx_convex_se(x, alpha, length, L, M, output_size, beta, name: str = ""):
    """
    Hilbert space approximation for the squared
    exponential kernel in the non-centered parametrisation for convex gp based on 
    https://drive.google.com/file/d/19kHhfuYC22ymNt3a3BlYgW0vWexOmbNy/view
    """
    
    phi = convex_eigenfunctions(x, L, M)
    spd = jnp.tile(jnp.sqrt(diag_spectral_density(alpha, length, L, M)), (output_size,1)).T
    weights = sample(name, beta, sample_shape=(M, output_size)) if isinstance(beta, Distribution) else beta ### can have multi-output
    gamma = spd * weights
    f = jnp.einsum("tj...,j... -> t..." , jnp.einsum("tjk,k... -> tj...", phi, gamma), gamma) ## output is length of x and output size
    return f