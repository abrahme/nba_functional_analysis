import jax.numpy as jnp 
import jax
from numpyro import sample
from numpyro.distributions import Distribution
import numpyro.distributions as dist


def spectral_density(w, alpha, length):
    D = length.shape[-1]
    c = alpha * jnp.power(2 * jnp.pi, D / 2) * jnp.prod(length)
    e = jnp.exp(-0.5 * (jnp.square(length) * jnp.square(w)).sum())
    return c * e

def sqrt_eigenvalues(M, L):
    D = L.shape[-1]
    return jnp.sqrt(jnp.tile(jnp.arange(1,  M + 1)[:, None], (1, D)) * jnp.pi / (2 * L) )

def make_convex_phi(x, L, M= 1):
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

def make_phi(x, L, M):
    eigen_funcs = eigenfunctions(x, L, M)
    return jnp.einsum("...m, ...k -> ...mk", eigen_funcs, eigen_funcs)

def diag_spectral_density(alpha, length, L, M):
    return spectral_density(sqrt_eigenvalues(M, L), alpha, length)


def eigenfunctions(x, L, M):
    """
    The first `M` eigenfunctions of the laplacian operator in `[-L, L] ^ D`
    evaluated at `x`. These are used for the approximation of the
    squared exponential kernel.
    """

    eig_vals = jnp.square(sqrt_eigenvalues(M, L))
    return jnp.prod(jnp.sin(jnp.einsum("n..., m... -> nm...", x + L, eig_vals)), -1) / jnp.prod(jnp.sqrt(L)) 


def make_gamma_phi_gamma(phi, gamma):
    right_result = jax.vmap(jax.vmap(lambda x: jnp.einsum("tjk,k -> tj", phi, x), -1, -1), -1, -1)(gamma)
    result = jax.vmap(jax.vmap(lambda x, y: jnp.einsum("tj,j -> t", x, y), -1, -1), -1, -1)(right_result, gamma)
     ## output is length of x and output size (mult by -1 to make sure we get concave functions)    
    return result


def make_gamma(weights, alpha, length, M, L, output_size):
    spd = jnp.moveaxis(jnp.tile(jnp.sqrt(diag_spectral_density(alpha, length, L, M)), (*output_size,1)), -1, 0)
    gamma = spd * weights
    return gamma

def make_psi_gamma(psi, gamma):
    return jnp.einsum("...m , nm -> n...", gamma, psi)

def make_convex_f(phi_x, psi_x, psi_x_time_cross, phi_time, shifted_x_time, L_time, M_time, alpha_time, length_time, weights_time, L, M, alpha, length, weights, slope, intercept, output_size):
    ## intercept should be n x k
    ### slope should be n x k 
    gamma_x = make_gamma(weights, alpha, length, M, L, output_size)
    gamma_time = make_gamma(weights_time, alpha_time, length_time, M_time, L_time, output_size)
    gamma_phi_gamma_x = make_gamma_phi_gamma(phi_x, gamma_x)

    return intercept + jnp.einsum("nk, t -> nkt",  slope, shifted_x_time) - jnp.einsum("..., t -> ...t" , gamma_phi_gamma_x, .5 * jnp.square(shifted_x_time)) - 2 * jnp.einsum("n..., ...t -> n...t",
                                                                                                                                                                           make_psi_gamma(psi_x, gamma_x),
                                                                                                                                                                           make_psi_gamma(psi_x_time_cross,
                                                                                                                                                                                          gamma_time)) - make_gamma_phi_gamma(phi_time, gamma_time) 
