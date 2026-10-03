"""
ComfyUI API Client.
Handles connection to remote ComfyUI, asset uploading (video & avatar),
workflow preparation, prompt submission, WebSocket/polling execution monitoring,
and output video retrieval.
"""

import os
import json
import time
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, Callable
from urllib.parse import urlparse, urljoin

import requests
import websocket

logger = logging.getLogger("comfy_client")

DEFAULT_COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "http://127.0.0.1:8188")
MIN_VALID_VIDEO_BYTES = 10 * 1024


class ComfyUIClient:
    def __init__(
        self,
        host: Optional[str] = None,
        template_path: Optional[Union[str, Path]] = None,
        avatars_dir: Optional[Union[str, Path]] = None,
    ):
        raw_host = host or DEFAULT_COMFYUI_HOST
        # Normalize host URL
        if not raw_host.startswith("http://") and not raw_host.startswith("https://"):
            raw_host = f"http://{raw_host}"
        self.host = raw_host.rstrip("/")

        if template_path is None:
            self.template_path = Path(__file__).resolve().parent.parent / "Framework_api.json"
        else:
            self.template_path = Path(template_path)

        if avatars_dir is None:
            self.avatars_dir = Path(__file__).resolve().parent.parent / "Avatars"
        else:
            self.avatars_dir = Path(avatars_dir)

    def _get_ws_url(self, client_id: str) -> str:
        parsed = urlparse(self.host)
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        netloc = parsed.netloc
        path = parsed.path.rstrip("/")
        return f"{ws_scheme}://{netloc}{path}/ws?clientId={client_id}"

    def check_health(self, timeout: float = 5.0) -> Dict[str, Any]:
        """Check ComfyUI server connectivity and retrieve system stats."""
        url = f"{self.host}/system_stats"
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                stats = resp.json()
                return {
                    "connected": True,
                    "host": self.host,
                    "stats": stats,
                }
            return {
                "connected": False,
                "host": self.host,
                "status_code": resp.status_code,
                "error": f"Unexpected status code {resp.status_code}",
            }
        except Exception as e:
            return {
                "connected": False,
                "host": self.host,
                "error": str(e),
            }

    def upload_file(
        self,
        file_path: Union[str, Path],
        subfolder: str = "",
        overwrite: bool = True,
        file_type: str = "input",
    ) -> Dict[str, Any]:
        """
        Upload file (video or image) to ComfyUI via POST /upload/image.
        Even for videos, ComfyUI saves into input/ subfolder via this multipart endpoint.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File to upload does not exist: {path}")

        url = f"{self.host}/upload/image"
        logger.info("Uploading %s to ComfyUI at %s", path.name, url)

        with open(path, "rb") as f:
            files = {
                "image": (path.name, f, "application/octet-stream"),
            }
            data = {
                "overwrite": "true" if overwrite else "false",
                "subfolder": subfolder,
                "type": file_type,
            }
            resp = requests.post(url, files=files, data=data, timeout=120)
            resp.raise_for_status()
            res_json = resp.json()
            logger.info("Uploaded %s successfully: %s", path.name, res_json)
            return res_json

    def upload_avatar_if_needed(self, avatar_name: str) -> str:
        """
        Ensures avatar file from Avatars/ is uploaded to ComfyUI.
        Returns the resolved avatar filename.
        """
        avatar_path = self.avatars_dir / avatar_name
        if not avatar_path.exists():
            # Check without extension or defaults
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = self.avatars_dir / f"{avatar_name}{ext}"
                if candidate.exists():
                    avatar_path = candidate
                    break

        if avatar_path.exists():
            res = self.upload_file(avatar_path, overwrite=True)
            return res.get("name", avatar_path.name)
        else:
            logger.warning("Avatar file %s not found in %s, using name as-is", avatar_name, self.avatars_dir)
            return avatar_name

    def load_workflow_template(self) -> Dict[str, Any]:
        """Load the API format workflow JSON."""
        if not self.template_path.exists():
            raise FileNotFoundError(f"Workflow API template not found at {self.template_path}")
        with open(self.template_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def prepare_prompt(
        self,
        video_filename: str,
        seed: Optional[int] = None,
        avatar_filename: Optional[str] = None,
        max_frames: Optional[int] = None,
        workflow_template: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Inject parameters into workflow graph nodes:
        - VHS_LoadVideo (Node 52 or class_type == VHS_LoadVideo): video input, optional frame_load_cap
        - KSampler (Node 353 or class_type == KSampler): seed
        - LoadImage (Node 167 or class_type == LoadImage): avatar image
        """
        prompt = json.loads(json.dumps(workflow_template or self.load_workflow_template()))

        vhs_nodes = []
        sampler_nodes = []
        image_nodes = []

        for node_id, node_data in prompt.items():
            class_type = node_data.get("class_type", "")
            if class_type == "VHS_LoadVideo":
                vhs_nodes.append((node_id, node_data))
            elif class_type == "KSampler":
                sampler_nodes.append((node_id, node_data))
            elif class_type == "LoadImage":
                image_nodes.append((node_id, node_data))

        # 1. Inject Video
        if "52" in prompt and prompt["52"].get("class_type") == "VHS_LoadVideo":
            prompt["52"]["inputs"]["video"] = video_filename
            if max_frames is not None and max_frames > 0:
                prompt["52"]["inputs"]["frame_load_cap"] = int(max_frames)
        elif vhs_nodes:
            node_id, node_data = vhs_nodes[0]
            node_data["inputs"]["video"] = video_filename
            if max_frames is not None and max_frames > 0:
                node_data["inputs"]["frame_load_cap"] = int(max_frames)
        else:
            raise KeyError("No VHS_LoadVideo node found in workflow template")

        # 2. Inject Seed
        if seed is not None:
            if "353" in prompt and prompt["353"].get("class_type") == "KSampler":
                prompt["353"]["inputs"]["seed"] = int(seed)
            elif sampler_nodes:
                sampler_nodes[0][1]["inputs"]["seed"] = int(seed)

        # 3. Inject Avatar
        if avatar_filename:
            if "167" in prompt and prompt["167"].get("class_type") == "LoadImage":
                prompt["167"]["inputs"]["image"] = avatar_filename
            elif image_nodes:
                image_nodes[0][1]["inputs"]["image"] = avatar_filename

        return prompt

    def submit_prompt(self, prompt: Dict[str, Any], client_id: str) -> str:
        """Submit prompt to POST /prompt and return prompt_id."""
        url = f"{self.host}/prompt"
        payload = {
            "prompt": prompt,
            "client_id": client_id,
        }
        resp = requests.post(url, json=payload, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to submit prompt to ComfyUI: HTTP {resp.status_code} - {resp.text}")

        res = resp.json()
        if "error" in res:
            raise RuntimeError(f"ComfyUI prompt error: {res['error']}")
        if res.get("node_errors"):
            raise RuntimeError(f"ComfyUI node validation errors: {res['node_errors']}")

        prompt_id = res.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"No prompt_id returned in ComfyUI response: {res}")

        logger.info("Prompt submitted successfully, prompt_id: %s", prompt_id)
        return prompt_id

    def wait_for_completion(
        self,
        prompt_id: str,
        client_id: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        poll_interval: float = 3.0,
        timeout_seconds: int = 1800,
    ) -> Dict[str, Any]:
        """
        Monitor prompt execution via WebSocket with automatic fallback to /history polling.
        Returns the history record for prompt_id upon completion.
        """
        start_time = time.time()
        ws_url = self._get_ws_url(client_id)
        ws = None

        try:
            ws = websocket.WebSocket()
            ws.settimeout(4.0)
            ws.connect(ws_url)
            logger.info("Connected to ComfyUI WebSocket: %s", ws_url)
        except Exception as e:
            logger.warning("Could not connect to ComfyUI WebSocket (%s). Falling back to HTTP polling.", e)
            ws = None

        try:
            while True:
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    raise TimeoutError(f"Workflow execution timed out after {timeout_seconds} seconds")

                # 1. Attempt reading WebSocket message if available
                ws_done = False
                if ws is not None:
                    try:
                        msg = ws.recv()
                        if isinstance(msg, str):
                            data = json.loads(msg)
                            msg_type = data.get("type")
                            msg_data = data.get("data", {})

                            if msg_type == "status":
                                if progress_callback:
                                    progress_callback({"stage": "queued", "data": msg_data})

                            elif msg_type == "progress":
                                if progress_callback:
                                    progress_callback({
                                        "stage": "sampling_progress",
                                        "value": msg_data.get("value"),
                                        "max": msg_data.get("max"),
                                    })

                            elif msg_type == "executing":
                                node = msg_data.get("node")
                                current_prompt_id = msg_data.get("prompt_id")
                                if current_prompt_id == prompt_id:
                                    if node is None:
                                        # Finished executing all nodes for this prompt!
                                        ws_done = True
                                    else:
                                        if progress_callback:
                                            progress_callback({"stage": "executing_node", "node": node})

                            elif msg_type == "execution_error":
                                if msg_data.get("prompt_id") == prompt_id:
                                    raise RuntimeError(f"ComfyUI execution error: {msg_data}")

                    except websocket.WebSocketTimeoutException:
                        # Normal socket timeout, fall through to poll history
                        pass
                    except Exception as e:
                        logger.warning("WebSocket recv error (%s). Falling back to HTTP polling.", e)
                        try:
                            ws.close()
                        except Exception:
                            pass
                        ws = None

                # 2. Check /history/{prompt_id}
                history = self.get_history(prompt_id)
                if prompt_id in history:
                    prompt_hist = history[prompt_id]
                    # Check if there is an execution error recorded in history
                    status_info = prompt_hist.get("status", {})
                    if status_info.get("status_str") == "error":
                        raise RuntimeError(f"ComfyUI job failed in history: {status_info}")
                    logger.info("Workflow completed according to ComfyUI history.")
                    return prompt_hist

                if ws_done:
                    # WebSocket announced completion, wait briefly for history to register
                    time.sleep(1.0)
                    history = self.get_history(prompt_id)
                    if prompt_id in history:
                        return history[prompt_id]

                time.sleep(poll_interval)

        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Fetch execution history for a given prompt_id."""
        url = f"{self.host}/history/{prompt_id}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning("Failed to query history for %s: %s", prompt_id, e)
        return {}

    def extract_output_video_info(self, history_data: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Extract output video file info from ComfyUI history outputs.
        Looks through all output nodes (especially VHS_VideoCombine or SaveVideoRGBA).
        Returns dict with keys: filename, subfolder, type
        """
        outputs = history_data.get("outputs", {})
        candidates = []

        for node_id, node_output in outputs.items():
            # Check gifs/videos/images arrays
            for key in ["gifs", "videos", "images"]:
                items = node_output.get(key, [])
                for item in items:
                    fname = item.get("filename", "")
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in [".mp4", ".webm", ".mov", ".gif"]:
                        candidates.append({
                            "node_id": node_id,
                            "filename": fname,
                            "subfolder": item.get("subfolder", ""),
                            "type": item.get("type", "output"),
                            "format": item.get("format", ""),
                        })

        if not candidates:
            return None

        # Prioritize WanAnimate prefix or node 226 if present
        for c in candidates:
            if c["node_id"] == "226" or "wananimate" in c["filename"].lower():
                return c

        return candidates[0]

    def download_output(
        self,
        filename: str,
        subfolder: str = "",
        output_type: str = "output",
        dest_path: Union[str, Path] = "output.mp4",
    ) -> Path:
        """Download output video file from ComfyUI via GET /view."""
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": output_type,
        }
        url = f"{self.host}/view"
        logger.info("Downloading synthesized output from %s with params %s to %s", url, params, dest)

        temp_dest = dest.with_suffix(".tmp")
        with requests.get(url, params=params, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(temp_dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)

        if not temp_dest.exists() or temp_dest.stat().st_size < MIN_VALID_VIDEO_BYTES:
            size = temp_dest.stat().st_size if temp_dest.exists() else 0
            if temp_dest.exists():
                temp_dest.unlink()
            raise IOError(f"Downloaded output video file is missing or too small ({size} bytes)")

        if dest.exists():
            dest.unlink()
        temp_dest.rename(dest)
        logger.info("Output video downloaded successfully to %s (%d bytes)", dest, dest.stat().st_size)
        return dest

    def run_pipeline(
        self,
        video_path: Union[str, Path],
        avatar_name: str = "avatar_nu.png",
        seed: int = 42,
        max_frames: Optional[int] = None,
        dest_path: Union[str, Path] = "output.mp4",
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        timeout_seconds: int = 1800,
    ) -> Path:
        """
        Executes end-to-end synthesis pipeline on ComfyUI:
        1. Uploads source video
        2. Uploads avatar
        3. Injects parameters into prompt template
        4. Submits prompt and waits for execution
        5. Downloads synthesized video to dest_path
        """
        client_id = uuid.uuid4().hex

        # 1. Upload source video
        if progress_callback:
            progress_callback({"stage": "uploading_video", "video": str(video_path)})
        upload_resp = self.upload_file(video_path)
        comfy_video_name = upload_resp.get("name", Path(video_path).name)

        # 2. Upload avatar
        if progress_callback:
            progress_callback({"stage": "uploading_avatar", "avatar": avatar_name})
        comfy_avatar_name = self.upload_avatar_if_needed(avatar_name)

        # 3. Prepare prompt
        prompt = self.prepare_prompt(
            video_filename=comfy_video_name,
            seed=seed,
            avatar_filename=comfy_avatar_name,
            max_frames=max_frames,
        )

        # 4. Submit & Wait
        if progress_callback:
            progress_callback({"stage": "submitting_prompt", "seed": seed})
        prompt_id = self.submit_prompt(prompt, client_id)

        history_record = self.wait_for_completion(
            prompt_id=prompt_id,
            client_id=client_id,
            progress_callback=progress_callback,
            timeout_seconds=timeout_seconds,
        )

        # 5. Extract output and download
        output_info = self.extract_output_video_info(history_record)
        if not output_info:
            raise RuntimeError(f"No output video file detected in ComfyUI history for prompt {prompt_id}")

        if progress_callback:
            progress_callback({"stage": "downloading_generated_video", "output_info": output_info})

        saved_path = self.download_output(
            filename=output_info["filename"],
            subfolder=output_info.get("subfolder", ""),
            output_type=output_info.get("type", "output"),
            dest_path=dest_path,
        )
        return saved_path
