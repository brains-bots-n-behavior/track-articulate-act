"""Video-driven articulated-joint estimation.

The package intentionally keeps measurement (RGB features and masks) separate
from the final one-degree-of-freedom motion model.  ``scripts/09_estimate_joint.py``
is the command-line entry point.
"""

from .fitting import JointFit, fit_joint_hypotheses

__all__ = ["JointFit", "fit_joint_hypotheses"]
