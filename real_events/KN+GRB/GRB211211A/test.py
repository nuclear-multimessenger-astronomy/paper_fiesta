from typing import List, Dict
import jax.numpy as jnp
import jax.lax as lax

def extract_key1(dicts: List[Dict[str, float]]):
    """
    Extracts values corresponding to "key1" from a list of dictionaries using jax.lax.scan.
    Assumes that all dictionaries contain "key1" with numeric values.
    """
    def scan_fn(carry, d):
        return carry, d.get("key1", 0.0)
    
    _, keys = lax.scan(scan_fn, None, dicts)
    return jnp.array(keys)

# Example usage:
data = [{"key1": 1.0, "key2": 2.0}, {"key1": 3.0, "key2": 4.0}]
result = extract_key1(data)
print(result)  # Output: [1. 3.]



