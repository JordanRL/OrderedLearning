"""Auto-discover experiment modules.

Each subdirectory is an experiment. Importing this package triggers
@ExperimentRegistry.register decorators in each experiment's __init__.py.
"""

import importlib
import pkgutil

for _, name, is_pkg in pkgutil.iter_modules(__path__):
    if is_pkg:
        importlib.import_module(f'.{name}', __package__)
