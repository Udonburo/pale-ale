"""Minimal synthetic-development implementation for Gate12C-2.

This package deliberately has no dependency on the retired Gate12C-2 control
plane.  Its public surface is the residual diagnostic and the development
runner/validator pair.
"""

from .metrics import ResidualDiagnostics, residual_diagnostics

__all__ = ["ResidualDiagnostics", "residual_diagnostics"]

