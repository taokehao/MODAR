"""Canonical CIF-like crystal language route.

This package is developed in parallel with the FlowMM/DiffCSP crystal-native
track. It must not modify or depend on FlowMM model code, configs, or outputs.
It reuses read-only multimodal data exported under
``data_prep/outputs/crystal_native_mp_wbm`` and writes its own artifacts under
``data_prep/outputs/crystal_language_mp_wbm``.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
