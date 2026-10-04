"""
Flask Evaluation Microservice for Vietnamese Sign Language (VSL) Synthesis.
Connects on-demand Kaggle dataset sampling, remote ComfyUI execution,
and standalone ViTPose/3D-ResNet kinematic & spatio-temporal evaluation.
"""

import os
import sys
import json
import uuid
import time
import queue
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from flask import Flask, request, jsonify, redirect, Response

# Add repo root to sys.path so modules can be imported consistently
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset_manager import KaggleDatasetManager
from src.comfy_client import ComfyUIClient, DEFAULT_COMFYUI_HOST

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vsl_eval_service")

# Lazy evaluation model cache
_EVAL_MODELS_LOCK = threading.Lock()
_RESNET_EXTRACTOR = None
_VITPOSE_DETECTOR = None


def get_eval_models(device: Optional[str] = None):
    """Lazy-load and cache ResNetVideoFeatureExtractor and ViTPoseWholeBodyDetector."""
    global _RESNET_EXTRACTOR, _VITPOSE_DETECTOR
    with _EVAL_MODELS_LOCK:
        if _RESNET_EXTRACTOR is None or _VITPOSE_DETECTOR is None:
            import torch
            from scripts.evaluate_benchmark import ResNetVideoFeatureExtractor, ViTPoseWholeBodyDetector

            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("Initializing benchmark evaluation models on device: %s", dev)
            if _RESNET_EXTRACTOR is None:
                _RESNET_EXTRACTOR = ResNetVideoFeatureExtractor(device=dev)
            if _VITPOSE_DETECTOR is None:
                _VITPOSE_DETECTOR = ViTPoseWholeBodyDetector(device=dev)

        return _RESNET_EXTRACTOR, _VITPOSE_DETECTOR


class JobManager:
    """Thread-safe background queue and job status manager."""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        self.output_dir = Path(output_dir or (REPO_ROOT / "outputs" / "jobs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._work_queue: queue.Queue = queue.Queue()
        self._dataset_mgr = KaggleDatasetManager()

        # Load previously saved jobs from disk
        self._load_persisted_jobs()

        # Single background worker thread to process jobs sequentially (RTX 5060 Ti 16GB safe)
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="JobWorker")
        self._worker_thread.start()
        logger.info("Background evaluation worker thread started.")

    def _load_persisted_jobs(self) -> None:
        """Loads existing job metadata from output_dir."""
        for job_file in self.output_dir.glob("*.json"):
            try:
                with open(job_file, "r", encoding="utf-8") as f:
                    job_data = json.load(f)
                    job_id = job_data.get("job_id")
                    if job_id:
                        # If service restarted while job was queued or running, mark as interrupted
                        if job_data.get("status") in ["queued", "running"]:
                            job_data["status"] = "failed"
                            job_data["error"] = "Service restarted while job was in progress"
                        self._jobs[job_id] = job_data
            except Exception as e:
                logger.warning("Could not read existing job file %s: %s", job_file, e)

    def _persist_job(self, job_id: str) -> None:
        """Save job state to JSON file."""
        job_data = self._jobs.get(job_id)
        if not job_data:
            return
        dest = self.output_dir / f"{job_id}.json"
        try:
            temp_dest = dest.with_suffix(".tmp")
            with open(temp_dest, "w", encoding="utf-8") as f:
                json.dump(job_data, f, indent=2, ensure_ascii=False)
            if dest.exists():
                dest.unlink()
            temp_dest.rename(dest)
        except Exception as e:
            logger.error("Failed to persist job %s: %s", job_id, e)

    def create_sample_job(
        self,
        n: int = 3,
        seed: int = 42,
        avatar: str = "avatar_nam.png",
        max_frames: Optional[int] = None,
        comfyui_host: Optional[str] = None,
        mock_comfyui: bool = False,
    ) -> Dict[str, Any]:
        """Create and queue a dataset sampling evaluation job."""
        samples = self._dataset_mgr.sample(n=n, seed=seed)
        if not samples:
            raise ValueError("Dataset contains no samples to evaluate.")

        eff_max_frames = int(max_frames) if (max_frames is not None and int(max_frames) > 0) else None
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        job = {
            "job_id": job_id,
            "job_type": "sample",
            "status": "queued",
            "seed": seed,
            "avatar": avatar,
            "max_frames": eff_max_frames,
            "comfyui_host": comfyui_host or DEFAULT_COMFYUI_HOST,
            "mock_comfyui": mock_comfyui,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "started_at": None,
            "completed_at": None,
            "total_samples": len(samples),
            "completed_samples": 0,
            "current_stage": "queued",
            "stage_details": {},
            "sampled_videos": samples,
            "results": [],
            "summary_metrics": {},
            "error": None,
        }

        with self._lock:
            self._jobs[job_id] = job
            self._persist_job(job_id)

        self._work_queue.put(job_id)
        logger.info("Queued sample evaluation job %s with %d samples", job_id, len(samples))
        return job

    def create_video_job(
        self,
        video_name: str,
        seed: int = 42,
        avatar: str = "avatar_nu.png",
        max_frames: Optional[int] = None,
        comfyui_host: Optional[str] = None,
        mock_comfyui: bool = False,
    ) -> Dict[str, Any]:
        """Create and queue a specific video evaluation job."""
        record = self._dataset_mgr.get_by_video_name(video_name)
        if not record:
            # Create a provisional record if not listed in CSV
            norm_name = video_name if video_name.endswith(".mp4") else f"{video_name}.mp4"
            record = {
                "id": "custom",
                "video": norm_name,
                "label": "unspecified",
                "is_cached": self._dataset_mgr.is_video_cached(norm_name),
            }

        eff_max_frames = int(max_frames) if (max_frames is not None and int(max_frames) > 0) else None
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        job = {
            "job_id": job_id,
            "job_type": "video",
            "status": "queued",
            "seed": seed,
            "avatar": avatar,
            "max_frames": eff_max_frames,
            "comfyui_host": comfyui_host or DEFAULT_COMFYUI_HOST,
            "mock_comfyui": mock_comfyui,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "started_at": None,
            "completed_at": None,
            "total_samples": 1,
            "completed_samples": 0,
            "current_stage": "queued",
            "stage_details": {},
            "sampled_videos": [record],
            "results": [],
            "summary_metrics": {},
            "error": None,
        }

        with self._lock:
            self._jobs[job_id] = job
            self._persist_job(job_id)

        self._work_queue.put(job_id)
        logger.info("Queued single video evaluation job %s for %s", job_id, record["video"])
        return job

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve job dictionary by ID."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                return dict(job)
        return None

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return list of most recent jobs."""
        with self._lock:
            jobs_list = sorted(
                self._jobs.values(),
                key=lambda j: j.get("created_at", ""),
                reverse=True,
            )
            return jobs_list[:limit]

    def re_evaluate_job(self, job_id: str) -> Dict[str, Any]:
        """
        Re-evaluates FVD and Kinematic metrics for all samples of an already generated job
        without calling ComfyUI synthesis again.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise KeyError(f"Job {job_id} not found")
            job["status"] = "running"
            job["current_stage"] = "evaluating_metrics"
            job["stage_details"] = {"status": "starting_re_evaluation"}
            self._persist_job(job_id)

        resnet_extractor, vitpose_detector = get_eval_models()
        from scripts.evaluate_benchmark import evaluate_pair

        results = job.get("results", [])
        job_dir = self.output_dir / job_id

        for idx, item in enumerate(results):
            v_name = item.get("video", "")
            base = os.path.splitext(v_name)[0]
            sp = item.get("source_path")
            gp = item.get("generated_path")

            if not sp or not os.path.exists(sp):
                try:
                    sp = str(self._dataset_mgr.download_video(v_name))
                    item["source_path"] = sp
                except Exception:
                    pass

            if not gp or not os.path.exists(gp):
                cand = job_dir / f"{base}_gen.mp4"
                if cand.exists():
                    gp = str(cand)
                    item["generated_path"] = gp

            if sp and gp and os.path.exists(sp) and os.path.exists(gp):
                with self._lock:
                    job["current_sample_index"] = idx + 1
                    job["current_sample_video"] = v_name
                    job["current_stage"] = "evaluating_metrics"
                    job["stage_details"] = {"video": v_name, "status": f"computing_fvd_and_kinematics ({idx+1}/{len(results)})"}
                    self._persist_job(job_id)

                try:
                    eval_out = evaluate_pair(
                        source_path=sp,
                        generated_path=gp,
                        resnet_extractor=resnet_extractor,
                        vitpose_detector=vitpose_detector,
                        max_frames=None,
                    )
                    item["frames"] = eval_out.get("frames")
                    item["FVD"] = eval_out.get("FVD")
                    item["kinematics"] = eval_out.get("kinematics_vitpose")
                except Exception as e:
                    logger.error("Failed re-evaluating pair %s vs %s: %s", sp, gp, e)

        with self._lock:
            job["status"] = "completed"
            job["current_stage"] = "completed"
            job["stage_details"] = {"status": "re_evaluation_finished"}
            job["completed_at"] = datetime.now(timezone.utc).isoformat()
            self._persist_job(job_id)

        logger.info("Successfully re-evaluated all samples for job %s without ComfyUI generation.", job_id)
        return dict(job)

    def _worker_loop(self) -> None:
        """Worker loop that sequentially pulls jobs from queue and executes them."""
        while True:
            job_id = self._work_queue.get()
            try:
                self._process_job(job_id)
            except Exception as e:
                logger.error("Unhandled error processing job %s: %s", job_id, e, exc_info=True)
                with self._lock:
                    if job_id in self._jobs:
                        self._jobs[job_id]["status"] = "failed"
                        self._jobs[job_id]["error"] = str(e)
                        self._jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
                        self._persist_job(job_id)
            finally:
                self._work_queue.task_done()

    def _process_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = "running"
            job["current_stage"] = "initializing_models"
            job["started_at"] = datetime.now(timezone.utc).isoformat()
            self._persist_job(job_id)

        logger.info("Started executing job %s", job_id)
        samples = job.get("sampled_videos", [])
        job_output_dir = self.output_dir / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)

        comfy_client = ComfyUIClient(host=job.get("comfyui_host"))
        resnet_extractor, vitpose_detector = get_eval_models()
        from scripts.evaluate_benchmark import evaluate_pair

        results = []

        for idx, sample in enumerate(samples):
            v_name = sample["video"]
            v_label = sample.get("label", "")
            base_name = os.path.splitext(v_name)[0]

            with self._lock:
                job["current_sample_index"] = idx + 1
                job["current_sample_video"] = v_name
                job["current_stage"] = "downloading_source"
                job["stage_details"] = {"video": v_name, "sample_number": idx + 1, "total": len(samples)}
                self._persist_job(job_id)

            # Step 1: Download / retrieve source video
            try:
                source_video_path = self._dataset_mgr.download_video(v_name)
            except Exception as e:
                logger.error("Failed downloading source video %s for job %s: %s", v_name, job_id, e)
                results.append({
                    "video": v_name,
                    "label": v_label,
                    "error": f"Failed downloading source video: {e}",
                })
                continue

            # Step 2: Run ComfyUI synthesis or mock mode
            with self._lock:
                job["current_stage"] = "generating_comfyui"
                job["stage_details"] = {"video": v_name, "status": "dispatching_prompt"}
                self._persist_job(job_id)

            gen_video_dest = job_output_dir / f"{base_name}_gen.mp4"

            if job.get("mock_comfyui", False):
                # Mock mode: copy or create test video for offline testing
                import shutil
                logger.info("Mock ComfyUI mode enabled for job %s, duplicating source video", job_id)
                time.sleep(0.5)
                shutil.copyfile(source_video_path, gen_video_dest)
            else:
                def progress_cb(info):
                    with self._lock:
                        job["stage_details"] = info
                        # Do not persist on every single step to avoid excessive disk I/O

                job_max_frames = job.get("max_frames")
                eff_max_frames = int(job_max_frames) if (job_max_frames is not None and int(job_max_frames) > 0) else None

                try:
                    comfy_client.run_pipeline(
                        video_path=source_video_path,
                        avatar_name=job.get("avatar", "avatar_nam.png"),
                        seed=job.get("seed", 42),
                        max_frames=eff_max_frames,
                        dest_path=gen_video_dest,
                        progress_callback=progress_cb,
                    )
                except Exception as e:
                    logger.error("ComfyUI synthesis failed for video %s: %s", v_name, e)
                    results.append({
                        "video": v_name,
                        "label": v_label,
                        "source_path": str(source_video_path),
                        "error": f"ComfyUI synthesis error: {e}",
                    })
                    continue

            # Step 3: Run benchmark evaluation
            with self._lock:
                job["current_stage"] = "evaluating_metrics"
                job["stage_details"] = {"video": v_name, "status": "computing_fvd_and_kinematics"}
                self._persist_job(job_id)

            try:
                eval_out = evaluate_pair(
                    source_path=str(source_video_path),
                    generated_path=str(gen_video_dest),
                    resnet_extractor=resnet_extractor,
                    vitpose_detector=vitpose_detector,
                    max_frames=eff_max_frames,
                )

                sample_result = {
                    "video": v_name,
                    "label": v_label,
                    "source_path": str(source_video_path),
                    "generated_path": str(gen_video_dest),
                    "frames": eval_out.get("frames"),
                    "FVD": eval_out.get("FVD"),
                    "kinematics": eval_out.get("kinematics_vitpose"),
                }
                results.append(sample_result)

            except Exception as e:
                logger.error("Benchmark evaluation failed for video %s: %s", v_name, e)
                results.append({
                    "video": v_name,
                    "label": v_label,
                    "source_path": str(source_video_path),
                    "generated_path": str(gen_video_dest),
                    "error": f"Metric computation error: {e}",
                })

            with self._lock:
                job["completed_samples"] = len(results)
                job["results"] = results
                self._persist_job(job_id)

        # Step 4: Finalize summary metrics
        summary_metrics = self._aggregate_summary(results)

        with self._lock:
            job["results"] = results
            job["summary_metrics"] = summary_metrics
            job["completed_at"] = datetime.now(timezone.utc).isoformat()
            job["status"] = "completed" if any("FVD" in r for r in results) else "failed"
            job["current_stage"] = "completed" if job["status"] == "completed" else "failed"
            self._persist_job(job_id)

        logger.info("Job %s finished with status: %s", job_id, job["status"])

    def _aggregate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregated averages across evaluated samples."""
        fvd_scores = [r["FVD"] for r in results if isinstance(r.get("FVD"), (int, float))]
        summary: Dict[str, Any] = {
            "total_evaluated": len(results),
            "successful_evaluations": len(fvd_scores),
        }

        if fvd_scores:
            summary["mean_FVD"] = round(sum(fvd_scores) / len(fvd_scores), 2)

        # Kinematics aggregation if available
        kin_results = [r.get("kinematics") for r in results if isinstance(r.get("kinematics"), dict)]
        if kin_results:
            parts = ["Overall", "Hands", "Hands_Shape", "Pose", "Face"]
            for part in parts:
                mpjpe_vals = []
                n_mpjpe_vals = []
                pck_05_vals = []
                pck_10_vals = []

                for k in kin_results:
                    part_metrics = k.get(part)
                    if isinstance(part_metrics, dict):
                        if "PA-MPJPE" in part_metrics:
                            mpjpe_vals.append(part_metrics["PA-MPJPE"])
                        if "N-PA-MPJPE" in part_metrics:
                            n_mpjpe_vals.append(part_metrics["N-PA-MPJPE"])
                        if "PA-PCK@0.05" in part_metrics:
                            pck_05_vals.append(part_metrics["PA-PCK@0.05"])
                        if "PA-PCK@0.10" in part_metrics:
                            pck_10_vals.append(part_metrics["PA-PCK@0.10"])

                if mpjpe_vals:
                    summary[f"mean_{part.lower()}_PA_MPJPE"] = round(sum(mpjpe_vals) / len(mpjpe_vals), 3)
                if n_mpjpe_vals:
                    summary[f"mean_{part.lower()}_N_PA_MPJPE"] = round(sum(n_mpjpe_vals) / len(n_mpjpe_vals), 2)
                if pck_05_vals:
                    summary[f"mean_{part.lower()}_PA_PCK@0.05"] = round(sum(pck_05_vals) / len(pck_05_vals), 2)
                if pck_10_vals:
                    summary[f"mean_{part.lower()}_PA_PCK@0.10"] = round(sum(pck_10_vals) / len(pck_10_vals), 2)
        else:
            summary["kinematics_note"] = "ViTPose kinematics omitted or pose models not loaded"

        return summary


OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "SignBridge 3D VSL ComfyUI Evaluation API",
        "description": "REST API microservice for Vietnamese Sign Language (VSL) video synthesis via remote ComfyUI (Wan 2.1 14B) and benchmark evaluation (3D ResNet-18 FVD & ViTPose WholeBody).",
        "version": "1.0.0",
    },
    "servers": [
        {"url": "/", "description": "Current host"}
    ],
    "paths": {
        "/api/v1/health": {
            "get": {
                "tags": ["System"],
                "summary": "Health Check",
                "description": "Check microservice health, GPU availability, and ComfyUI server status.",
                "parameters": [
                    {
                        "name": "comfyui_host",
                        "in": "query",
                        "required": False,
                        "schema": {"type": "string"},
                        "description": "Optional ComfyUI host URL to probe (defaults to COMFYUI_HOST env var)"
                    }
                ],
                "responses": {
                    "200": {"description": "Service health and connection details"}
                }
            }
        },
        "/api/v1/dataset/info": {
            "get": {
                "tags": ["Dataset"],
                "summary": "Dataset Statistics",
                "description": "Get Kaggle VSL dataset metadata and local video cache statistics.",
                "responses": {
                    "200": {"description": "Dataset metadata statistics"}
                }
            }
        },
        "/api/v1/dataset/search": {
            "get": {
                "tags": ["Dataset"],
                "summary": "Search Dataset",
                "description": "Search sign language videos by gloss label or filename.",
                "parameters": [
                    {
                        "name": "q",
                        "in": "query",
                        "required": False,
                        "schema": {"type": "string"},
                        "description": "Search keyword (e.g. 'cá voi', 'địa chỉ')"
                    },
                    {
                        "name": "limit",
                        "in": "query",
                        "required": False,
                        "schema": {"type": "integer", "default": 20},
                        "description": "Max results to return"
                    }
                ],
                "responses": {
                    "200": {"description": "Matching dataset entries"}
                }
            }
        },
        "/api/v1/evaluate/sample": {
            "post": {
                "tags": ["Evaluation"],
                "summary": "Queue Dataset Sampling Evaluation",
                "description": "Sample N videos deterministically from Kaggle, synthesize avatar animations via ComfyUI, and compute benchmark metrics.",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "n": {"type": "integer", "default": 3, "description": "Number of videos to sample (1-100)"},
                                    "seed": {"type": "integer", "default": 42, "description": "Sampling and generation seed"},
                                    "avatar": {"type": "string", "default": "avatar_nam.png", "description": "Reference avatar filename from Avatars/"},
                                    "max_frames": {"type": "integer", "nullable": True, "description": "Max frames to generate and evaluate (null for all)"},
                                    "comfyui_host": {"type": "string", "nullable": True, "description": "Override ComfyUI host URL"},
                                    "mock_comfyui": {"type": "boolean", "default": False, "description": "Enable mock generation for testing without ComfyUI"}
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "202": {"description": "Evaluation job queued successfully"}
                }
            }
        },
        "/api/v1/evaluate/video": {
            "post": {
                "tags": ["Evaluation"],
                "summary": "Queue Single Video Evaluation",
                "description": "Download or use a specific video, synthesize avatar animations via ComfyUI, and compute benchmark metrics.",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["video_name"],
                                "properties": {
                                    "video_name": {"type": "string", "example": "D0001B.mp4", "description": "Kaggle video filename"},
                                    "seed": {"type": "integer", "default": 42, "description": "Sampling and generation seed"},
                                    "avatar": {"type": "string", "default": "avatar_nu.png", "description": "Reference avatar filename from Avatars/"},
                                    "max_frames": {"type": "integer", "nullable": True, "description": "Max frames to generate and evaluate"},
                                    "comfyui_host": {"type": "string", "nullable": True, "description": "Override ComfyUI host URL"},
                                    "mock_comfyui": {"type": "boolean", "default": False, "description": "Enable mock generation for testing without ComfyUI"}
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "202": {"description": "Evaluation job queued successfully"}
                }
            }
        },
        "/api/v1/jobs/{job_id}": {
            "get": {
                "tags": ["Jobs"],
                "summary": "Get Job Status & Results",
                "description": "Retrieve execution status, current stage, progress, and computed benchmark metrics.",
                "parameters": [
                    {
                        "name": "job_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Job identifier (e.g. job_20261003_135527_559ba7)"
                    }
                ],
                "responses": {
                    "200": {"description": "Job details, real-time stage, and evaluation results"},
                    "404": {"description": "Job not found"}
                }
            }
        },
        "/api/v1/jobs/{job_id}/re-evaluate": {
            "post": {
                "tags": ["Jobs"],
                "summary": "Re-evaluate Existing Job Videos",
                "description": "Re-calculate FVD and ViTPose Kinematic metrics on already generated videos without re-running ComfyUI synthesis.",
                "parameters": [
                    {
                        "name": "job_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Job identifier (e.g. job_20261003_220956_382aa5)"
                    }
                ],
                "responses": {
                    "202": {"description": "Re-evaluation started in background"},
                    "404": {"description": "Job not found"}
                }
            }
        },
        "/api/v1/jobs": {
            "get": {
                "tags": ["Jobs"],
                "summary": "List Recent Jobs",
                "description": "Retrieve list of all recent evaluation jobs and their statuses.",
                "parameters": [
                    {
                        "name": "limit",
                        "in": "query",
                        "required": False,
                        "schema": {"type": "integer", "default": 20},
                        "description": "Max jobs to return"
                    }
                ],
                "responses": {
                    "200": {"description": "List of recent evaluation jobs"}
                }
            }
        }
    }
}

SWAGGER_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SignBridge 3D VSL ComfyUI Evaluation API - Swagger UI</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui.css" />
  <style>
    body { margin: 0; background: #fafafa; }
    .topbar { display: none; }
    .swagger-ui .info { margin: 25px 0; }
  </style>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui-bundle.js" crossorigin></script>
  <script>
    window.onload = () => {
      window.ui = SwaggerUIBundle({
        url: '/api/v1/openapi.json',
        dom_id: '#swagger-ui',
        deepLinking: true,
        presets: [
          SwaggerUIBundle.presets.apis,
          SwaggerUIBundle.SwaggerUIStandalonePreset
        ],
        layout: "BaseLayout"
      });
    };
  </script>
</body>
</html>
"""


def create_app(job_manager: Optional[JobManager] = None) -> Flask:
    """Flask application factory."""
    app = Flask(__name__)
    mgr = job_manager or JobManager()
    dataset_mgr = KaggleDatasetManager()

    # Enable CORS for all routes
    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        return response

    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"status": "error", "error": str(e)}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"status": "error", "error": "Resource not found"}), 404

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"status": "error", "error": "Internal server error"}), 500

    @app.route("/", methods=["GET"])
    def root():
        """Redirect root to Swagger documentation UI."""
        return redirect("/docs")

    @app.route("/docs", methods=["GET"])
    @app.route("/swagger", methods=["GET"])
    def swagger_ui():
        """Serve Swagger UI documentation page."""
        return Response(SWAGGER_UI_HTML, mimetype="text/html")

    @app.route("/api/v1/openapi.json", methods=["GET"])
    def openapi_spec():
        """Serve OpenAPI 3.0 JSON specification."""
        return jsonify(OPENAPI_SPEC)

    @app.route("/api/v1/health", methods=["GET"])
    def health_check():
        """Check microservice health and ComfyUI server status."""
        comfy_host = request.args.get("comfyui_host", DEFAULT_COMFYUI_HOST)
        client = ComfyUIClient(host=comfy_host)
        comfy_status = client.check_health(timeout=3.0)

        import torch
        return jsonify({
            "status": "healthy",
            "service": "signbridge-comfyui-evaluation",
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "comfyui": comfy_status,
            "dataset": dataset_mgr.get_dataset_info(),
        }), 200

    @app.route("/api/v1/dataset/info", methods=["GET"])
    def dataset_info():
        """Get dataset metadata and cache statistics."""
        try:
            info = dataset_mgr.get_dataset_info()
            return jsonify({"status": "success", "dataset": info}), 200
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/api/v1/dataset/search", methods=["GET"])
    def dataset_search():
        """Search available dataset entries by label or video name."""
        query = request.args.get("q", "").strip()
        try:
            limit = int(request.args.get("limit", 20))
        except ValueError:
            limit = 20

        try:
            results = dataset_mgr.search(query=query, limit=limit)
            return jsonify({
                "status": "success",
                "query": query,
                "count": len(results),
                "results": results,
            }), 200
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/api/v1/evaluate/sample", methods=["POST"])
    def evaluate_sample():
        """
        Sample N items from Kaggle dataset, run ComfyUI generation & benchmark.
        Request body: {"n": 3, "seed": 42, "avatar": "avatar_nam.png", "max_frames": null, "comfyui_host": null}
        """
        data = request.get_json(silent=True) or {}
        try:
            n = int(data.get("n", 3))
            if n <= 0 or n > 100:
                return jsonify({"status": "error", "error": "n must be between 1 and 100"}), 400
        except ValueError:
            return jsonify({"status": "error", "error": "n must be an integer"}), 400

        seed = int(data.get("seed", 42))
        avatar = str(data.get("avatar") or "avatar_nam.png")
        max_frames_raw = data.get("max_frames")
        max_frames = None
        if max_frames_raw is not None:
            try:
                val = int(max_frames_raw)
                if val > 0:
                    max_frames = val
            except (ValueError, TypeError):
                max_frames = None
        comfyui_host = data.get("comfyui_host")
        mock_comfyui = bool(data.get("mock_comfyui", False))

        try:
            job = mgr.create_sample_job(
                n=n,
                seed=seed,
                avatar=avatar,
                max_frames=max_frames,
                comfyui_host=comfyui_host,
                mock_comfyui=mock_comfyui,
            )
            return jsonify({
                "status": "success",
                "job_id": job["job_id"],
                "message": f"Evaluation job queued for {n} samples with seed {seed}",
                "sampled_videos": job["sampled_videos"],
                "poll_url": f"/api/v1/jobs/{job['job_id']}",
            }), 202
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/api/v1/evaluate/video", methods=["POST"])
    def evaluate_video():
        """
        Run pipeline for a specific video filename.
        Request body: {"video_name": "D0001N.mp4", "seed": 42, "avatar": "avatar_nu.png", "max_frames": null}
        """
        data = request.get_json(silent=True) or {}
        video_name = data.get("video_name")
        if not video_name or not isinstance(video_name, str):
            return jsonify({"status": "error", "error": "video_name is required"}), 400

        seed = int(data.get("seed", 42))
        avatar = str(data.get("avatar") or "avatar_nu.png")
        max_frames_raw = data.get("max_frames")
        max_frames = None
        if max_frames_raw is not None:
            try:
                val = int(max_frames_raw)
                if val > 0:
                    max_frames = val
            except (ValueError, TypeError):
                max_frames = None
        comfyui_host = data.get("comfyui_host")
        mock_comfyui = bool(data.get("mock_comfyui", False))

        try:
            job = mgr.create_video_job(
                video_name=video_name,
                seed=seed,
                avatar=avatar,
                max_frames=max_frames,
                comfyui_host=comfyui_host,
                mock_comfyui=mock_comfyui,
            )
            sample = job["sampled_videos"][0]
            return jsonify({
                "status": "success",
                "job_id": job["job_id"],
                "message": f"Evaluation job queued for video {sample['video']} with seed {seed}",
                "video": sample["video"],
                "label": sample.get("label", ""),
                "poll_url": f"/api/v1/jobs/{job['job_id']}",
            }), 202
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/api/v1/jobs/<job_id>", methods=["GET"])
    def get_job_status(job_id: str):
        """Retrieve real-time status and metrics of a job."""
        job = mgr.get_job(job_id)
        if not job:
            return jsonify({"status": "error", "error": f"Job {job_id} not found"}), 404
        return jsonify(job), 200

    @app.route("/api/v1/jobs/<job_id>/re-evaluate", methods=["POST"])
    def re_evaluate_job(job_id: str):
        """Re-evaluate metrics for all generated videos of an existing job without running ComfyUI."""
        job = mgr.get_job(job_id)
        if not job:
            return jsonify({"status": "error", "error": f"Job {job_id} not found"}), 404

        threading.Thread(target=mgr.re_evaluate_job, args=(job_id,), daemon=True).start()
        return jsonify({
            "status": "success",
            "job_id": job_id,
            "message": f"Re-evaluation started in background for job {job_id}",
            "poll_url": f"/api/v1/jobs/{job_id}",
        }), 202

    @app.route("/api/v1/jobs", methods=["GET"])
    def list_jobs():
        """List recent evaluation jobs."""
        try:
            limit = int(request.args.get("limit", 20))
        except ValueError:
            limit = 20
        jobs = mgr.list_jobs(limit=limit)
        return jsonify({
            "status": "success",
            "count": len(jobs),
            "jobs": jobs,
        }), 200

    return app


app = create_app()

if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", 5000))
    logger.info("Starting VSL Evaluation Microservice on %s:%d", host, port)
    app.run(host=host, port=port, debug=False, threaded=True)
