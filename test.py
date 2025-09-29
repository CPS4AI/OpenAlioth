import jax
import jax.numpy as jnp

from woe import DDSketch, percentile

u, v = jnp.arange(12).reshape(4, 3), jnp.arange(15, 15+15).reshape(5, 3)
x = jnp.vstack((u, v))
beta = jnp.array([0.25, 0.5])
print(f"plaintext quantile: {percentile(u, v, beta)}")


dds = DDSketch(0.02, beta)
tmp = x.copy()
tmp = tmp.at[tmp == 0].set(1)
min_abs_value = jnp.min(jnp.abs(tmp), axis=0)
max_abs_value = jnp.max(jnp.abs(tmp), axis=0)
B, offset = dds.update_params_from_range(min_abs_value, max_abs_value)
print(f"encrypted quantile: {dds.ddsketch(u, v)}")