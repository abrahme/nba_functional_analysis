import jax.numpy as jnp 
import jax
from jaxlib.xla_extension import ArrayImpl
from typing import get_args

def eigenindices(m: list[int] | int, dim: int) -> ArrayImpl:
    """Returns the indices of the first :math:`D \\times m^\\star` eigenvalues of the laplacian operator.

    .. math::

        m^\\star = \\prod_{i=1}^D m_i

    For more details see Eq. (10) in [1].

    **References:**

        1. Riutort-Mayol, G., Bürkner, PC., Andersen, M.R. et al. Practical Hilbert space
           approximate Bayesian Gaussian processes for probabilistic programming. Stat Comput 33, 17 (2023).

    :param list[int] | int m: The number of desired eigenvalue indices in each dimension.
        If an integer, the same number of eigenvalues is computed in each dimension.
    :param int dim: The dimension of the space.

    :returns: An array of the indices of the first :math:`D \\times m^\\star` eigenvalues.
    :rtype: ArrayImpl

    **Examples:**

    .. code-block:: python

            >>> import jax.numpy as jnp

            >>> from numpyro.contrib.hsgp.laplacian import eigenindices

            >>> m = 10
            >>> S = eigenindices(m, 1)
            >>> assert S.shape == (1, m)
            >>> S
            Array([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]], dtype=int32)

            >>> m = 10
            >>> S = eigenindices(m, 2)
            >>> assert S.shape == (2, 100)

            >>> m = [2, 2, 3]  # Riutort-Mayol et al eq (10)
            >>> S = eigenindices(m, 3)
            >>> assert S.shape == (3, 12)
            >>> S
            Array([[1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
                   [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
                   [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]], dtype=int32)

    """
    if isinstance(m, int):
        m = [m] * dim
    elif len(m) != dim:
        raise ValueError("The length of m must be equal to the dimension of the space.")
    return (
        jnp.stack(
            jnp.meshgrid(*[jnp.arange(1, m_ + 1) for m_ in m], indexing="ij"), axis=-1
        )
        .reshape(-1, dim)
        .T
    )





def sqrt_eigenvalues(
    ell: int | float | list[int | float], m: list[int] | int, dim: int
) -> ArrayImpl:
    """
    The first :math:`m^\\star \\times D` square root of eigenvalues of the laplacian operator in
    :math:`[-L_1, L_1] \\times ... \\times [-L_D, L_D]`. See Eq. (56) in [1].

    **References:**

        1. Solin, A., Särkkä, S. Hilbert space methods for reduced-rank Gaussian process regression.
           Stat Comput 30, 419-446 (2020)

    :param int | float | list[int | float] ell: The length of the interval in each dimension divided by 2.
        If a float, the same length is used in each dimension.
    :param list[int] | int m: The number of eigenvalues to compute in each dimension.
        If an integer, the same number of eigenvalues is computed in each dimension.
    :param int dim: The dimension of the space.

    :returns: An array of the first :math:`m^\\star \\times D` square root of eigenvalues.
    :rtype: ArrayImpl
    """
    S = eigenindices(m, dim)
    return S * jnp.pi / 2 / ell # dim x prod(m) array of eigenvalues



def eigenfunctions(
    x: ArrayImpl, ell: float | list[float], m: int | list[int]
) -> ArrayImpl:
    """
    The first :math:`m^\\star` eigenfunctions of the laplacian operator in
    :math:`[-L_1, L_1] \\times ... \\times [-L_D, L_D]`
    evaluated at values of `x`. See Eq. (56) in [1].
    If `x` is 1D, the problem is assumed unidimensional.
    Otherwise, the dimension of the input space is inferred as the size of the last dimension of
    `x`. Other dimensions are treated as batch dimensions.

    **Example:**

    .. code-block:: python

        >>> import jax.numpy as jnp

        >>> from numpyro.contrib.hsgp.laplacian import eigenfunctions

        >>> n = 100
        >>> m = 10

        >>> x = jnp.linspace(-1, 1, n)

        >>> basis = eigenfunctions(x=x, ell=1.2, m=m)

        >>> assert basis.shape == (n, m)

        >>> x = jnp.ones((n, 3))  # 2d input
        >>> basis = eigenfunctions(x=x, ell=1.2, m=[2, 2, 3])
        >>> assert basis.shape == (n, 12)


    **References:**

        1. Solin, A., Särkkä, S. Hilbert space methods for reduced-rank Gaussian process regression.
           Stat Comput 30, 419-446 (2020)

    :param ArrayImpl x: The points at which to evaluate the eigenfunctions.
        If `x` is 1D the problem is assumed unidimensional.
        Otherwise, the dimension of the input space is inferred as the last dimension of `x`.
        Other dimensions are treated as batch dimensions.
    :param float | list[float] ell: The length of the interval in each dimension divided by 2.
        If a float, the same length is used in each dimension.
    :param int | list[int] m: The number of eigenvalues to compute in each dimension.
        If an integer, the same number of eigenvalues is computed in each dimension.
    :returns: An array of the first :math:`m^\\star \\times D` eigenfunctions evaluated at `x`.
    :rtype: ArrayImpl
    """
    if x.ndim == 1:
        x_ = x[..., None]
    else:
        x_ = x
    dim = x_.shape[-1]  # others assumed batch dims
    n_batch_dims = x_.ndim - 1
    a = jnp.expand_dims(ell, tuple(range(n_batch_dims)))
    b = jnp.expand_dims(sqrt_eigenvalues(ell, m, dim), tuple(range(n_batch_dims)))
    return jnp.prod(jnp.sqrt(1 / a) * jnp.sin(b * (x_[..., None] + a)), axis=-2)





def spectral_density( w: ArrayImpl, alpha: float, length: float | ArrayImpl
) -> float:
    """
    Spectral density of the squared exponential kernel.

    See Section 4.2 in [1] and Section 2.1 in [2].

    .. math::

        S(\\boldsymbol{\\omega}) = \\alpha (\\sqrt{2\\pi})^D \\ell^D
            \\exp\\left(-\\frac{1}{2} \\ell^2 \\boldsymbol{\\omega}^{T} \\boldsymbol{\\omega}\\right)


    **References:**

        1. Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning.

        2. Riutort-Mayol, G., Bürkner, PC., Andersen, M.R. et al. Practical Hilbert space
           approximate Bayesian Gaussian processes for probabilistic programming. Stat Comput 33, 17 (2023).

    :param int dim: dimension
    :param ArrayImpl w: frequency
    :param float alpha: amplitude
    :param float length: length scale
    :return: spectral density value
    :rtype: float
    """
    c = alpha * jnp.prod(jnp.sqrt(2 * jnp.pi) * length, axis=-1)
    e = jnp.exp(-0.5 * jnp.sum(w**2 * length**2, axis=-1))
    return c * e


def diag_spectral_density(
    dim: int,
    alpha: float,
    length: float | list[float],
    ell: float | int | list[float | int],
    m: int | list[int],
) -> ArrayImpl:
    """
    Evaluates the spectral density of the squared exponential kernel at the first :math:`D \\times m^\\star`
    square root eigenvalues of the laplacian operator in :math:`[-L_1, L_1] \\times ... \\times [-L_D, L_D]`.

    :param float alpha: amplitude of the squared exponential kernel
    :param float length: length scale of the squared exponential kernel
    :param float | int | list[float | int] ell: The length of the interval divided by 2 in each dimension.
        If a float or int, the same length is used in each dimension.
    :param int | list[int] m: The number of eigenvalues to compute for each dimension.
        If an integer, the same number of eigenvalues is computed in each dimension.
    :param int dim: The dimension of the space

    :return: spectral density vector evaluated at the first :math:`D \\times m^\\star` square root eigenvalues
    :rtype: ArrayImpl
    """

    def _spectral_density(w):
        return spectral_density(w=w, alpha=alpha, length=length
        )

    sqrt_eigenvalues_ = sqrt_eigenvalues(ell=ell, m=m, dim = dim)  # dim x m
    return jax.vmap(_spectral_density, in_axes=-1)(sqrt_eigenvalues_)

# def spectral_density(w, alpha, length):
#     D = length.shape[-1]
#     c = alpha * jnp.power(2 * jnp.pi, D / 2) * jnp.prod(length)
#     e = jnp.exp(-0.5 * (jnp.square(length * w)).sum(-1))
#     return c * e

# def sqrt_eigenvalues(M, L):
#     D = L.shape[-1]
#     return jnp.sqrt(jnp.tile(jnp.arange(1,  M + 1)[:, None], (1, D)) * jnp.pi / (2 * L) ).sum(-1, keepdims=True)

def make_convex_phi(x, L, M= 1):
    assert len(x.shape) == 1 ### only have capacity for single dimension concavity
    eig_vals = jnp.squeeze(sqrt_eigenvalues(L, M, 1))
    broadcast_sub = jax.vmap(jax.vmap(jnp.subtract, (None, 0)), (0, None))
    broadcast_add = jax.vmap(jax.vmap(jnp.add, (None, 0)), (0, None))
    sum_eig_vals = broadcast_add(eig_vals, eig_vals)
    diff_eig_vals = broadcast_sub(eig_vals, eig_vals)
    diff_eig_vals_square = jnp.power( diff_eig_vals, 2)
    sum_eig_vals_square = jnp.power(sum_eig_vals, 2)
    x_shifted = x + L
    cos_pos = jnp.cos(jnp.einsum("t,m... -> tm...", x_shifted, sum_eig_vals)) / (2 * L * sum_eig_vals_square)
    cos_neg = jnp.cos(jnp.einsum("t,m... -> tm...", x_shifted, diff_eig_vals)) / (2 * L * diff_eig_vals_square)
    diagonal_constants = broadcast_sub(jnp.square(x - L)/ (4 * L) - L , (1 / (2 * L * jnp.diagonal(sum_eig_vals_square)))) ### should be t x m
    diagonal_elements = diagonal_constants + jnp.diagonal(cos_pos, axis1=1, axis2=2)
    other_elements = cos_pos - cos_neg 
    broadcast_fill_diag = jax.vmap(lambda x, y: jnp.fill_diagonal(x,y, inplace=False), in_axes = 0)
    phi = broadcast_fill_diag(other_elements, diagonal_elements)
    return phi #should be t x m x m where t is the length of x and m is the number of eigen values (or M)



# def diag_spectral_density(alpha, length, L, M):
#     return spectral_density(sqrt_eigenvalues(M, L), alpha, length)


# def eigenfunctions(x, L, M):
#     """
#     The first `M` eigenfunctions of the laplacian operator in `[-L, L] ^ D`
#     evaluated at `x`. These are used for the approximation of the
#     squared exponential kernel.
#     """

#     eig_vals = jnp.square(sqrt_eigenvalues(M, L))
#     return jnp.prod(jnp.sin(jnp.einsum("n..., m... -> nm...", x + L, eig_vals)), -1) / jnp.prod(jnp.sqrt(L)) 


def make_gamma_phi_gamma(phi, gamma):
    right_result = jnp.einsum("njk,k... -> nj...", phi, gamma)
    result = jnp.einsum("nj..., j... -> n...",right_result, gamma)
     ## output is length of x and output size (mult by -1 to make sure we get concave functions)    
    return result


def make_gamma(weights, alpha, length, M, L, output_size, dim):

    spd = jnp.sqrt(diag_spectral_density(dim, alpha, length, L, M))
    if spd.shape[-1] != 1:
        spd_ = jnp.expand_dims(spd, -1)
    else:
        spd_ = spd
    gamma = spd_ * weights
    return gamma

def make_psi_gamma(psi, gamma):
    return jnp.dot(psi,gamma)

def make_convex_f(phi_x, psi_x, psi_x_time_cross, phi_time, shifted_x_time, L_time, M_time, alpha_time, length_time, weights_time, weights, slope, intercept, output_size):
    ## intercept should be n x k
    ### slope should be n x k 
    gamma_time = make_gamma(weights_time, alpha_time, length_time, M_time, L_time, output_size, 1)
    gamma_phi_gamma_x = make_gamma_phi_gamma(phi_x, weights)
    psi_gamma_x = make_psi_gamma(psi_x, weights)
    psi_gamma_cross = make_psi_gamma(psi_x_time_cross, gamma_time)
    parabolic = jnp.einsum("..., t -> ...t" , gamma_phi_gamma_x, .5 * jnp.square(shifted_x_time))
    cross_term = 2 * jnp.einsum("n..., t... -> n...t",psi_gamma_x,psi_gamma_cross)
    return jnp.swapaxes(intercept + jnp.einsum("nk, t -> nkt",  slope, shifted_x_time) - parabolic - cross_term - make_gamma_phi_gamma(phi_time, gamma_time).T[None,...] ,0,1)
