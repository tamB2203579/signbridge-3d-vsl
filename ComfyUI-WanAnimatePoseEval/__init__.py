"""
ComfyUI-WanAnimatePoseEval
Quantitative pose-alignment evaluation (MPJPE + PCK) for the WanAnimate
sign-language generation workflow.

Provides two nodes:
  - PoseEvaluationMetrics : re-extracts keypoints from generated frames and
                            compares against ground-truth pose_data.
  - SavePoseMetrics       : writes the metrics to JSON and CSV in the ComfyUI
                            output folder.
"""

from .nodes import PoseEvaluationMetrics, SavePoseMetrics

NODE_CLASS_MAPPINGS = {
    "PoseEvaluationMetrics": PoseEvaluationMetrics,
    "SavePoseMetrics": SavePoseMetrics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseEvaluationMetrics": "Pose Evaluation Metrics (PCK / MPJPE)",
    "SavePoseMetrics": "Save Pose Metrics (JSON/CSV)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]