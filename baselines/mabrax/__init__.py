"""The five continuous-control MaBrax tasks evaluated in the paper."""

import jax

# Brax 0.10.3's MJCF loader calls the alias removed in JAX 0.6. Restore the
# identical function only when this optional environment package is imported.
if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree.map
