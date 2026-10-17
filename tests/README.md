# Tests

Tests use small in-memory values and synthetic fixtures only. They do not use
clinical images, datasets, model weights, credentials, or a GPU. `conftest.py`
blocks socket connections during test collection and execution.
