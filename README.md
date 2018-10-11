Sample Contractions
-------------------

This is a collection of large sample tensor network contractions for testing ``opt_einsum`` with.


Usage
-----

E.g. to load file ``'oe_sample_MAXCUT_n50_reg5_seed0.json'``:

```python
import json

with open('oe_sample_MAXCUT_n50_reg5_seed0.json') as file:
    loaded = json.load(file)
    eq, shapes = loaded['eq'], loaded['shapes']
```