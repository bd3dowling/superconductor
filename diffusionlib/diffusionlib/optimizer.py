from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np
import particles.distributions as dists
from jax import random
from jaxtyping import Array, PRNGKeyArray
from particles.core import SMC
from particles.state_space_models import Bootstrap, StateSpaceModel

from diffusionlib.sampler import DDIMVP


@dataclass
class SMCDiffOptOptimizer:
    base_sampler: DDIMVP
    gamma_t: Callable[[int], float]
    num_particles: int = 1000

    def optimize(
        self,
        rng: PRNGKeyArray,
        f: Callable[[Array], Array],
        stack_samples: bool = False,
        **kwargs: Any,
    ) -> Array:
        rng, sub_rng = random.split(rng)
        np.random.seed(sub_rng[0])

        ssm = self.SSM(self.base_sampler)
        fk = self.PF(f, self.gamma_t, ssm, data=jnp.zeros(self.base_sampler.num_steps))
        smc = SMC(fk=fk, N=self.num_particles, store_history=True, **kwargs)
        smc.run()

        return smc.hist.X if stack_samples else smc.X

    class SSM(StateSpaceModel):
        def __init__(self, sampler: DDIMVP, **kwargs: Any):
            super().__init__(**kwargs)
            self.sampler = sampler
            self.dim_x = sampler.shape[1:]
            self.ts = self.sampler.ts[::-1]

        def PX0(self):
            return dists.MvNormal(loc=jnp.zeros(self.dim_x), cov=jnp.eye(*self.dim_x))

        def PX(self, t, xp):
            vec_t = jnp.full(xp.shape[0], self.ts[t - 1])
            x_mean, std = self.sampler.posterior(xp, vec_t)
            scale = std[0]  # NOTE: std identical for all particles
            return dists.MvNormal(loc=x_mean, scale=scale, cov=jnp.eye(*self.dim_x))

    class PF(Bootstrap):
        def __init__(
            self, f: Callable[[Array], Array], gamma_t: Callable[[int], float], ssm=None, data=None
        ):
            super().__init__(ssm, data)
            self.f = f
            self.gamma_t = gamma_t

        def logG(self, t, xp, x):
            if t == 0:
                return np.array(-self.gamma_t(self.T - t) * self.f(x))

            g_y_t = -self.gamma_t(self.T - t) * self.f(x)
            g_y_t1 = -self.gamma_t(self.T - (t - 1)) * self.f(xp)

            return np.array(g_y_t - g_y_t1)
