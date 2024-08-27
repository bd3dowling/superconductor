# Fixes from using np > 1.20
import numpy as np

setattr(np, "int", int)
setattr(np, "float", float)
setattr(np, "bool", bool)

import functools as ft
from pathlib import Path
from typing import List

import design_bench
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.random as random
import matplotlib.pyplot as plt
import optax  # NOTE: You need to manually patch the package, changing tuple -> Tuple
import pandas as pd
from design_bench.task import Task
from jaxtyping import Array, Float, Integer, PRNGKeyArray
from sklearn.model_selection import train_test_split

from diffusionlib.optimizer import SMCDiffOptOptimizer
from diffusionlib.sampler import DDIMVP

SEED = 100

T = jnp.array(1000)
BETA_MIN = jnp.array(0.1) / T
BETA_MAX = jnp.array(20) / T
LEARNING_RATE = 3e-4
NUM_STEPS = 40_000
BATCH_SIZE = 256
NUM_SAMPLES = 100


# Define network, loss, and training loop


class FullyConnectedWithTime(eqx.Module):
    """A simple model with multiple fully connected layers and some fourier features for the time
    variable.
    """

    layers: List[eqx.nn.Linear]

    def __init__(self, in_size: int, key: PRNGKeyArray):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        out_size = in_size

        self.layers = [
            eqx.nn.Linear(in_size + 4, 256, key=key1),
            eqx.nn.Linear(256, 256, key=key2),
            eqx.nn.Linear(256, 256, key=key3),
            eqx.nn.Linear(256, out_size, key=key4),
        ]

    def __call__(self, x: Array, t: Array) -> Array:
        t_fourier = jnp.array(
            [t - 0.5, jnp.cos(2 * jnp.pi * t), jnp.sin(2 * jnp.pi * t), -jnp.cos(4 * jnp.pi * t)],
        )

        x = jnp.concatenate([x, t_fourier])

        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))

        x = self.layers[-1](x)

        return x


@jax.jit
@jax.value_and_grad
def loss(model: FullyConnectedWithTime, data: Array, key: PRNGKeyArray) -> Array:
    key1, key2 = random.split(key, 2)

    random_times = random.randint(key1, (data.shape[0],), minval=0, maxval=T)

    # NOTE: noise will match as both use key2
    noise = random.normal(key2, data.shape)
    noised_data = forward_marginal(key2, data, random_times[:, jnp.newaxis])

    # NOTE: rescale time to in [0, 1]
    output = jax.vmap(model)(noised_data, random_times / (T - 1))

    loss = jnp.mean((noise - output) ** 2)

    return loss


def single_loss_fn(model, data, t, key):
    noise = random.normal(key, data.shape)
    noised_data = forward_marginal(key, data, t)

    output = model(noised_data, t / (T - 1))

    return jnp.mean((noise - output) ** 2)


def batch_loss_fn(model, data, key):
    batch_size = data.shape[0]
    t_key, loss_key = jr.split(key)
    loss_key = jr.split(loss_key, batch_size)

    # Low-discrepancy sampling over t to reduce variance
    t = random.randint(t_key, (batch_size,), minval=0, maxval=T)

    loss_fn = ft.partial(single_loss_fn, model)
    loss_fn = jax.vmap(loss_fn)

    return jnp.mean(loss_fn(data, t, loss_key))


def dataloader(data, batch_size, *, key):
    dataset_size = data.shape[0]
    indices = jnp.arange(dataset_size)
    while True:
        key, subkey = jr.split(key, 2)
        perm = jr.permutation(subkey, indices)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield data[batch_perm]
            start = end
            end = start + batch_size


@eqx.filter_jit
def make_step(model, data, key, opt_state, opt_update):
    loss_fn = eqx.filter_value_and_grad(batch_loss_fn)
    loss, grads = loss_fn(model, data, key)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    key = jr.split(key, 1)[0]
    return loss, model, key, opt_state


def get_y_max():
    # Load data
    data_path = Path().absolute() / "design-bench" / "design_bench_data" / "superconductor"

    y_files = sorted(data_path.glob("*y*.npy"))
    y_data = jnp.vstack([jnp.load(file) for file in y_files])

    return y_data.max()


# Define the diffusion
def beta_t(
    t: Float[Array, " batch"],
    beta_min: Float[Array, ""] = BETA_MIN,
    beta_max: Float[Array, ""] = BETA_MAX,
) -> Float[Array, " batch"]:
    return beta_min + t * (beta_max - beta_min) / T


def alpha_t(t: Float[Array, " batch"]) -> Float[Array, " batch"]:
    return 1 - beta_t(t)


alpha = alpha_t(jnp.arange(T + 1))
cumulative_alpha_values = jnp.cumprod(alpha)


def c_t(t: Integer[Array, " batch"]) -> Float[Array, " batch"]:
    return jnp.sqrt(cumulative_alpha_values[t])


def d_t(t: Integer[Array, " batch"]) -> Float[Array, " batch"]:
    return jnp.sqrt(1 - cumulative_alpha_values[t])


def forward_marginal(
    key: PRNGKeyArray, x_0: Float[Array, "batch dim"], t: Integer[Array, " batch"]
) -> Float[Array, "batch dim"]:
    return c_t(t) * x_0 + d_t(t) ** 2 * random.normal(key, x_0.shape)


def train_model(
    key: PRNGKeyArray,
    task: Task,
    num_steps: int,
    lr: float,
    batch_size: int,
    print_every: int = 1000,
):
    split_key, model_key, train_key, loader_key = jr.split(key, 4)

    # Train-Test split
    # NOTE: we don't use the y for training; just want associated correctly
    train_x, val_x, train_y, val_y = train_test_split(
        task.x, task.y, test_size=0.1, random_state=int(split_key[0])
    )

    data = task.normalize_x(train_x)
    val_data = task.normalize_x(val_x)

    model = FullyConnectedWithTime(data.shape[1], key=model_key)

    opt = optax.adabelief(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    total_value = 0
    total_size = 0
    for step, data_ in zip(range(num_steps), dataloader(data, batch_size, key=loader_key)):
        value, model, train_key, opt_state = make_step(
            model, data_, train_key, opt_state, opt.update
        )
        total_value += value.item()
        total_size += 1
        if (step % print_every) == 0 or step == num_steps - 1:
            key, sub_key = jr.split(key)
            val_loss = batch_loss_fn(model, val_data, sub_key)

            print(
                f"Step={step:05}",
                f"Train Loss={total_value / total_size:.4f}",
                f"Val Loss={val_loss:.4f}",
                sep="\t|\t",
            )

            total_value = 0
            total_size = 0

    return model, train_x, val_x, train_y, val_y


def plot_and_save(data, file_name):
    indices = np.arange(data.shape[0])
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.axhline(0, color="black", lw=0.8)
    ax.stem(indices, data, linefmt="C0-", markerfmt="C0o", basefmt=" ")

    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.set_xlim(-1, len(indices))
    ax.set_ylabel("Value")

    textstr = f"Temperature = {task.predict(data[np.newaxis, :])[0][0]:.2f}"
    ax.text(
        0.95,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
    )

    plt.tight_layout()
    plt.savefig(f"{file_name}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    key = jr.PRNGKey(SEED)

    task = design_bench.make("Superconductor-RandomForest-v0")
    dim_x = task.x.shape[1]  # 86

    print("Training...")
    key, sub_key = jr.split(key)
    model, train_x, val_x, train_y, val_y = train_model(
        sub_key, task, NUM_STEPS, LEARNING_RATE, BATCH_SIZE
    )

    print("Unconditional sampling...")
    sampler = DDIMVP(
        num_steps=T,
        shape=(NUM_SAMPLES, dim_x),
        model=jax.vmap(model),  # assumes epsilon model (not score), so okay here!
        beta_min=BETA_MIN,
        beta_max=BETA_MAX,
        eta=1.0,  # NOTE: equates to using DDPM
    )

    unconditional_samples = sampler.sample(key)

    print("Optimizing...")
    optimizer = SMCDiffOptOptimizer(
        base_sampler=sampler, gamma_t=lambda t: 1 - d_t(t), num_particles=NUM_SAMPLES
    )
    particle_samples = optimizer.optimize(key, lambda x: -task.predict(task.denormalize_x(x)))

    particle_y = task.predict(task.denormalize_x(particle_samples))
    best_y = jnp.max(particle_y)

    print("Saving plots...")
    plot_and_save(task.x[100], "real-sample")
    plot_and_save(
        jnp.clip(task.denormalize_x(unconditional_samples), a_min=0)[0], "unconditional-sample"
    )
    plot_and_save(
        jnp.clip(task.denormalize_x(particle_samples), a_min=0)[0], "optimised-sample"
    )
