import jax.numpy as jnp 
import jax
from jax.scipy.stats import norm
from jaxlib.xla_extension import ArrayImpl
from typing import get_args
from numpyro.distributions import Categorical, Independent, Normal, MixtureSameFamily



def make_spectral_mixture_density(eigenvalue, mu, cov, mixture_weight):
    """
    eigenvalue: (N, 5)
    mu: (16, 5, K)
    cov: (16, 5, K)   # diagonal covariances
    mixture_weight: (16, K)
    
    returns: (16, N) log probs of all N eigenvalues under each of 16 mixtures
    """
    def log_prob_single_prior(mu_b, cov_b, weight_b):
        """
        mu_b: (5, K)
        cov_b: (5, K)
        weight_b: (K,)
        """
        # Mixture distribution
        mixing_dist = Categorical(probs=weight_b)  # (K,)
        # Each component is 5-D multivariate with diagonal covariance
        component_dist = Independent(Normal(mu_b, cov_b), 1)  # event_shape=5
        mixture = MixtureSameFamily(mixing_dist, component_dist)
        # eigenvalue: (N, 5)
        return mixture.log_prob(eigenvalue.T)  # returns (N,)
    
    # vmap over the 16 priors
    logp = jax.vmap(log_prob_single_prior, in_axes=(0, 0, 0))(mu, cov, mixture_weight)
    return jnp.sqrt(jnp.exp(logp)) 

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



# @jax.custom_vjp
# def safe_cos_term(divisor, x, L):
#     denom = 2 * L * jnp.square(divisor)
#     z = jnp.where(divisor == 0, 0, divisor * (x+L))
#     numer = 1 - jnp.cos(z)
#     return jnp.where(z == 0, jnp.square(x + L)/ (4 * L), numer / denom)


# @jax.custom_vjp
# def safe_sin_term(divisor, x, L):
#     denom = 2 * L * divisor
#     z = jnp.where(divisor == 0, 0, divisor * (x+L))
#     numer = jnp.sin(z)
#     return jnp.where(z == 0, (x+L) / (2*L), numer / denom)



# def safe_cos_term_fwd(divisor, x, L):
#     y = safe_cos_term(divisor, x, L)
#     return y, (divisor, x, L)

# def safe_cos_term_bwd(res, g):
#     divisor, x, L = res
#     # derivative w.r.t. x is safe_sin_term
#     grad_x = safe_sin_term(divisor, x, L)
#     # compute partial derivatives for all inputs
#     return (  # returns tuple of same length as inputs
#         jnp.zeros_like(divisor),  # ∂y/∂divisor (assume 0)
#         jnp.sum(g * grad_x),                # ∂y/∂x
#         jnp.zeros_like(L)        # ∂y/∂L (assume 0)
#     )

# def safe_sin_term_fwd(divisor, x, L):
#     y = safe_sin_term(divisor, x, L)
#     return y, (divisor, x, L)

# def safe_sin_term_bwd(res, g):
#     divisor, x, L = res

#     grad_x = jnp.where(divisor == 0, 1 / (2 * L), jnp.cos(divisor * (x + L)) / (2 * L))
#     return (
#         jnp.zeros_like(divisor),  # ∂y/∂divisor (assume 0)
#         jnp.sum(g * grad_x),                # ∂y/∂x
#         jnp.zeros_like(L)
#     )

# safe_cos_term.defvjp(safe_cos_term_fwd, safe_cos_term_bwd)
# safe_cos_term = jax.jit(safe_cos_term)
# safe_sin_term.defvjp(safe_sin_term_fwd, safe_sin_term_bwd)
# safe_sin_term = jax.jit(safe_sin_term)


@jax.jit
def safe_cos_term(divisor, x, L, eps=1e-5):
    divisor_safe = jnp.where(jnp.abs(divisor) < eps, eps, divisor)
    z = divisor_safe * (x + L)
    denom = 2 * L * jnp.square(divisor_safe)
    numer = 1 - jnp.cos(z)
    
    base = numer / denom
    limit = jnp.square(x + L) / (4 * L)

    blend_weight = jnp.exp(-jnp.square(z / eps))

    return blend_weight * limit + (1 - blend_weight) * base

@jax.jit
def safe_sin_term(divisor, x, L, eps=1e-5):
    divisor_safe = jnp.where(jnp.abs(divisor) < eps, eps, divisor)
    z = divisor_safe * (x + L)
    denom = 2 * L * divisor_safe
    numer = jnp.sin(z)
    
    base = numer / denom
    limit = (x + L) / (2 * L)

    blend_weight = jnp.exp(-jnp.square(z / eps))

    return blend_weight * limit + (1 - blend_weight) * base


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


    a = ell
    b = sqrt_eigenvalues(ell, m, 1)
    return jnp.sqrt(1 / a) * jnp.sin(b * (x[..., None] + a))


def eigenfunctions_multivariate(x: ArrayImpl, ell: float | list[float], m: int | list[int]) -> ArrayImpl:
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

    :param ArrayLike x: The points at which to evaluate the eigenfunctions.
        If `x` is 1D the problem is assumed unidimensional.
        Otherwise, the dimension of the input space is inferred as the last dimension of `x`.
        Other dimensions are treated as batch dimensions.
    :param float | list[float] ell: The length of the interval in each dimension divided by 2.
        If a float, the same length is used in each dimension.
    :param int | list[int] m: The number of eigenvalues to compute in each dimension.
        If an integer, the same number of eigenvalues is computed in each dimension.
    :returns: An array of the first :math:`m^\\star \\times D` eigenfunctions evaluated at `x`.
    :rtype: Array
    """

    dim = x.shape[-1]  # others assumed batch dims
    n_batch_dims = jnp.ndim(x) - 1
   
    a = jnp.expand_dims(ell, tuple(range(n_batch_dims)))
    b = jnp.expand_dims(sqrt_eigenvalues(ell, m, dim), tuple(range(n_batch_dims)))
    return jnp.prod(
        jnp.sqrt(1 / a) * jnp.sin(b * (jnp.expand_dims(x, axis=-1) + a)), axis=-2
    )

def eigenfunctions_deriv(
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

    a = ell
    b = sqrt_eigenvalues(ell, m, 1)
    return jnp.prod(b * jnp.sqrt(1/ a) * jnp.cos(b * (x[..., None] + a)), axis=-2)



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
    # jax.debug.print("this is the lengthscale inside the spectral density function :{length}", length = length)
    # jax.debug.print("this is the alpha inside the spectral density function :{alpha}", alpha = alpha)
    c = alpha * jnp.prod(jnp.sqrt(2 * jnp.pi) * length, axis=-1)
    # jax.debug.print("this is the omega inside the spectral density function :{omega}", omega = w)
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


def make_convex_phi(x, L, M= 1):
    eig_vals = jnp.squeeze(sqrt_eigenvalues(L, M, 1))
    sum_eig_vals = eig_vals[None] + eig_vals[..., None]
    diff_eig_vals = eig_vals[None] - eig_vals[..., None]
    cos_pos  = safe_cos_term(sum_eig_vals, x, L)
    cos_neg = safe_cos_term(diff_eig_vals, x, L)
    phi = cos_neg - cos_pos
    return phi
    
def make_convex_phi_prime(x, L, M = 1):
    eig_vals = jnp.squeeze(sqrt_eigenvalues(L, M, 1))
    sum_eig_vals = eig_vals[None] + eig_vals[..., None]
    diff_eig_vals = eig_vals[None] - eig_vals[..., None]
    sin_pos = safe_sin_term(sum_eig_vals, x, L)
    sin_neg = safe_sin_term(diff_eig_vals, x, L)
    phi_prime = sin_neg - sin_pos
    return phi_prime #should be t x m x m where t is the length of x and m is the number of eigen values (or M)

def make_convex_phi_double_prime(x, L, M = 1):
    eigenfunction_vals = eigenfunctions(x, L, M)
    phi_double_prime = jnp.outer(eigenfunction_vals, eigenfunction_vals)
    return phi_double_prime #should be t x m x m where t is the length of x and m is the number of eigen values (or M)

def make_convex_phi_triple_prime(x, L, M = 1):

    eigenfunction_vals = eigenfunctions(x, L, M)
    deriv_eigenfunction_vals = eigenfunctions_deriv(x,L,M)
    term = jnp.outer(eigenfunction_vals, deriv_eigenfunction_vals)
    phi_triple_prime = term + term.T
    return phi_triple_prime #should be t x m x m where t is the length of x and m is the number of eigen values (or M)

def vmap_make_convex_phi_triple_prime(x, L, M):
    return jax.vmap(lambda t: make_convex_phi_triple_prime(t, L, M))(x)

def vmap_make_convex_phi_double_prime(x, L, M):
    return jax.vmap(lambda t: make_convex_phi_double_prime(t, L, M))(x)

def vmap_make_convex_phi_prime(x, L, M):
    return jax.vmap(lambda t: make_convex_phi_prime(t, L, M))(x)

def vmap_make_convex_phi(x, L, M):
    return jax.vmap(lambda t: make_convex_phi(t, L, M))(x)

def make_psi_gamma(psi, gamma):
    return jnp.einsum("nm, m... -> n...", psi,gamma)

def make_convex_f(gamma_phi_gamma_time, shifted_x_time, slope, intercept):
    ## intercept should be n x k
    ### slope should be n x k 

    return jnp.swapaxes(intercept + jnp.einsum("nk, t -> nkt", slope, shifted_x_time) - gamma_phi_gamma_time, 0,1)



