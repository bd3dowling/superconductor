"""Utility functions, including all functions related to
loss computation, optimization and sampling.
"""
import jax.numpy as jnp
import jax.random as random
from jax import vmap
from jaxtyping import Array


def get_timestep(t, t0, t1, num_steps):
    return (jnp.rint((t - t0) * (num_steps - 1) / (t1 - t0))).astype(jnp.int32)


def continuous_to_discrete(betas, dt):
    discrete_betas = betas * dt
    return discrete_betas


def get_sigma_function(sigma_min, sigma_max):
    log_sigma_min = jnp.log(sigma_min)
    log_sigma_max = jnp.log(sigma_max)

    def sigma(t):
        # return sigma_min * (sigma_max / sigma_min)**t  # Has large relative error close to zero compared to alternative, below
        return jnp.exp(log_sigma_min + t * (log_sigma_max - log_sigma_min))

    return sigma


def get_linear_beta_function(beta_min, beta_max):
    """Returns:
    Linear beta (cooling rate parameter) as a function of time,
    It's integral multiplied by -0.5, which is the log mean coefficient of the VP SDE.
    """

    def beta(t):
        return beta_min + t * (beta_max - beta_min)

    def log_mean_coeff(t):
        """..math: -0.5 * \\int_{0}^{t} \beta(t) dt"""
        return -0.5 * t * beta_min - 0.25 * t**2 * (beta_max - beta_min)

    return beta, log_mean_coeff


def get_cosine_beta_function(offset=0.08):
    """Returns:
    Squared cosine beta (cooling rate parameter) as a function of time,
    It's integral multiplied by -0.5, which is the log mean coefficient of the VP SDE.
    """

    def beta(t):
        # return jnp.cos((1. - t + offset) / (1 + offset) * 0.5 * jnp.pi)**2
        # Use double angle formula here, instead
        return 0.5 * (jnp.cos((1.0 - t + offset) / (1.0 + offset) * jnp.pi) + 1.0)

    def log_mean_coeff(t):
        """..math: -0.5 * \\int_{0}^{t} \beta(t) dt"""
        return -1.0 / 4 * (t - (1.0 + offset) * jnp.sin(jnp.pi * t / (1.0 + offset)) / jnp.pi)

    return beta, log_mean_coeff


def get_times(num_steps=1000, dt=None, t0=None):
    r"""
    Get linear, monotonically increasing time schedule.
    Args:
        num_steps: number of discretization time steps.
        dt: time step duration, float or `None`.
          Optional, if provided then final time, t1 = dt * num_steps.
        t0: A small float 0. < t0 << 1. The SDE or ODE are integrated to
            t0 to avoid numerical issues.
    Return:
        ts: JAX array of monotonically increasing values t \in [t0, t1].
    """
    if dt is not None:
        if t0 is not None:
            t1 = dt * (num_steps - 1) + t0
            # Defined in forward time, t \in [t0, t1], 0 < t0 << t1
            ts, step = jnp.linspace(t0, t1, num_steps, retstep=True)
            ts = ts.reshape(-1, 1)
            assert jnp.isclose(step, (t1 - t0) / (num_steps - 1))
            assert jnp.isclose(step, dt)
            dt = step
            assert t0 == ts[0]
        else:
            t1 = dt * num_steps
            # Defined in forward time, t \in [dt , t1], 0 < \t0 << t1
            ts, step = jnp.linspace(0.0, t1, num_steps + 1, retstep=True)
            ts = ts[1:].reshape(-1, 1)
            assert jnp.isclose(step, dt)
            dt = step
            t0 = ts[0]
    else:
        t1 = 1.0
        if t0 is not None:
            ts, dt = jnp.linspace(t0, 1.0, num_steps, retstep=True)
            ts = ts.reshape(-1, 1)
            assert jnp.isclose(dt, (1.0 - t0) / (num_steps - 1))
            assert t0 == ts[0]
        else:
            # Defined in forward time, t \in [dt, 1.0], 0 < dt << 1
            ts, dt = jnp.linspace(0.0, 1.0, num_steps + 1, retstep=True)
            ts = ts[1:].reshape(-1, 1)
            assert jnp.isclose(dt, 1.0 / num_steps)
            t0 = ts[0]
    assert ts[0, 0] == t0
    assert ts[-1, 0] == t1
    dts = jnp.diff(ts)
    assert jnp.all(dts > 0.0)
    assert jnp.all(dts == dt)
    return ts, dt


def batch_linalg_solve_A(A, b):
    return vmap(lambda b: jnp.linalg.solve(A, b))(b)


def batch_linalg_solve(A, b):
    return vmap(jnp.linalg.solve)(A, b)


def batch_mul(a: Array, b: Array) -> Array:
    return vmap(lambda a, b: a * b)(a, b)


def batch_mul_A(a, b):
    return vmap(lambda b: a * b)(b)


def batch_matmul(A, b):
    return vmap(lambda A, b: A @ b)(A, b)


def batch_matmul_A(A, b):
    return vmap(lambda b: A @ b)(b)


def errors(t, sde, score, rng, data, likelihood_weighting=True):
    """
    Args:
      ts: JAX array of times.
      sde: Instantiation of a valid SDE class.
      score: A function taking in (x, t) and returning the score.
      rng: Random number generator from JAX.
      data: A batch of samples from the training data, representing samples from the data distribution, shape (J, N).
      likelihood_weighting: Bool, set to `True` if likelihood weighting, as described in Song et al. 2020 (https://arxiv.org/abs/2011.13456), is applied.
    Returns:
      A Monte-Carlo approximation to the (likelihood weighted) score errors.
    """
    m = sde.mean_coeff(t)
    mean = batch_mul(m, data)
    std = jnp.sqrt(sde.variance(t))
    rng, step_rng = random.split(rng)
    noise = random.normal(step_rng, data.shape)
    x = mean + batch_mul(std, noise)
    if not likelihood_weighting:
        return noise + batch_mul(score(x, t), std)
    else:
        return batch_mul(noise, 1.0 / std) + score(x, t)


def get_loss(
    sde,
    solver,
    model,
    score_scaling=True,
    likelihood_weighting=True,
    reduce_mean=True,
    pointwise_t=False,
):
    """Create a loss function for score matching training.
    Args:
      sde: Instantiation of a valid SDE class.
      solver: Instantiation of a valid Solver class.
      model: A valid flax neural network `:class:flax.linen.Module` class.
      score_scaling: Bool, set to `True` if learning a score scaled by the marginal standard deviation.
      likelihood_weighting: Bool, set to `True` if likelihood weighting, as described in Song et al. 2020 (https://arxiv.org/abs/2011.13456), is applied.
      reduce_mean: Bool, set to `True` if taking the mean of the errors in the loss, set to `False` if taking the sum.
      pointwise_t: Bool, set to `True` if returning a function that can evaluate the loss pointwise over time. Set to `False` if returns an expectation of the loss over time.

    Returns:
      A loss function that can be used for score matching training.
    """
    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
    if pointwise_t:

        def pointwise_loss(t, params, rng, data):
            n_batch = data.shape[0]
            ts = jnp.ones((n_batch,)) * t
            score = get_score(sde, model, params, score_scaling)
            e = errors(ts, sde, score, rng, data, likelihood_weighting)
            losses = e**2
            losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
            if likelihood_weighting:
                g2 = sde.sde(jnp.zeros_like(data), ts)[1] ** 2
                losses = losses * g2
            return jnp.mean(losses)

        return pointwise_loss
    else:

        def loss(params, rng, data):
            rng, step_rng = random.split(rng)
            ts = random.uniform(step_rng, (data.shape[0],), minval=solver.ts[0], maxval=solver.t1)
            score = get_score(sde, model, params, score_scaling)
            e = errors(ts, sde, score, rng, data, likelihood_weighting)
            losses = e**2
            losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
            if likelihood_weighting:
                g2 = sde.sde(jnp.zeros_like(data), ts)[1] ** 2
                losses = losses * g2
            return jnp.mean(losses)

        return loss


def get_score(sde, model, params, score_scaling):
    if score_scaling is True:
        return lambda x, t: -batch_mul(model.apply(params, x, t), 1.0 / jnp.sqrt(sde.variance(t)))
    else:
        return lambda x, t: -model.apply(params, x, t)


def get_epsilon(sde, model, params, score_scaling):
    if score_scaling is True:
        return lambda x, t: model.apply(params, x, t)
    else:
        return lambda x, t: batch_mul(jnp.sqrt(sde.variance(t)), model.apply(params, x, t))


def shared_update(rng, x, t, solver, probability_flow=None):
    """A wrapper that configures and returns the update function of the solvers.

    :probablity_flow: Placeholder for probability flow ODE (TODO).
    """
    return solver.update(rng, x, t)


def sample_sphere(num_points: int) -> Array:
    alphas = jnp.linspace(0, 2 * jnp.pi * (1 - 1 / num_points), num_points)
    xs = jnp.cos(alphas)
    ys = jnp.sin(alphas)

    return jnp.stack([xs, ys], axis=1)
