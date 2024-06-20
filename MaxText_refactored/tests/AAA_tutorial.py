import jax
import jax.numpy as jnp
import flax.linen as nn

print(type(jax.devices()[0]))
print(len(jax.devices()))

d = 512


# alternative:
key1, key2 = jax.random.split(jax.random.key(0))


q_key = jax.random.key(42)
k_key = jax.random.key(0)

q = jax.random.uniform(q_key, (1, d), minval=1, maxval=100)
k = jax.random.uniform(k_key, (1, d), minval=1, maxval=100)

x = jnp.arange(start=0, stop=20)

shape = (1, 20)

x = jnp.reshape(x, shape=shape)
first, second = jnp.split(x, 2, -1)

print(first.shape)
print(second.shape)
print(first)
print(second)

test_object = nn.Module

print(dir(test_object))