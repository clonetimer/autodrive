"""Training-only expert controllers and research distillation utilities."""

from .metadrive_idm import MetaDriveIDMTeacher
from .distillation import AnchoredExpertDistillationResult, distill_anchored_expert

__all__ = [
    "MetaDriveIDMTeacher",
    "AnchoredExpertDistillationResult",
    "distill_anchored_expert",
]
