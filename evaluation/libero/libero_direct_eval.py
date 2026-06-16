#!/usr/bin/env python3
"""
Direct LIBERO evaluation for X-VLA checkpoints.

This script keeps the LIBERO rollout behavior from libero_client.py, but loads
the XVLA model in-process and calls model.generate_actions(...) directly. It
does not start or contact a FastAPI server.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence


def _print_help_without_runtime_deps() -> None:
    parser = argparse.ArgumentParser("Direct LIBERO evaluation for X-VLA")
    parser.add_argument("--model_path", required=True, help="XVLA checkpoint path or Hugging Face id")
    parser.add_argument(
        "--processor_path",
        default=None,
        help="Processor path or id; defaults to --model_path",
    )
    parser.add_argument("--lora_path", default=None, help="Optional LoRA adapter path")
    parser.add_argument(
        "--adapter_mode",
        default="single",
        choices=["single", "fuse", "suite"],
        help="Adapter routing mode. 'suite' chooses adapter folder from task suite.",
    )
    parser.add_argument(
        "--suite_lora_root",
        default=None,
        help="HF repo id or local root containing goal/object/long/spatial adapter folders.",
    )
    parser.add_argument(
        "--suite_lora_map",
        default=None,
        help="Optional suite=folder overrides, e.g. libero_10=long,libero_goal=goal.",
    )
    parser.add_argument(
        "--suite_lora_checkpoint_subdir",
        default=None,
        help="Optional checkpoint subfolder appended inside each suite folder.",
    )
    parser.add_argument(
        "--fuse_first_lora_path",
        default=None,
        help="First adapter for fuse mode; defaults to --lora_path.",
    )
    parser.add_argument(
        "--fuse_rest_lora_path",
        default=None,
        help="Adapter used after --fuse_switch_step in fuse mode.",
    )
    parser.add_argument(
        "--fuse_switch_step",
        type=int,
        default=20,
        help="LIBERO env step where fuse mode switches to the rest adapter.",
    )
    parser.add_argument(
        "--task_suites",
        nargs="+",
        default=["libero_10", "libero_spatial", "libero_goal", "libero_object"],
        help="LIBERO suite names; accepts space-separated or comma-separated values",
    )
    parser.add_argument(
        "--task_ids",
        default=None,
        help="Task ids to evaluate, e.g. 0,1,4-7. Defaults to all tasks.",
    )
    parser.add_argument("--eval_time", type=int, default=10, help="Episodes per selected task")
    parser.add_argument("--init_seed", type=int, default=42, help="Fallback seed for env and policy")
    parser.add_argument(
        "--env_seed",
        type=int,
        default=None,
        help="LIBERO environment seed; defaults to --init_seed.",
    )
    parser.add_argument(
        "--policy_seed",
        type=int,
        default=None,
        help="Python/NumPy/Torch/CUDA policy seed; defaults to --init_seed.",
    )
    parser.add_argument("--device", default="cuda", help="Device: cuda, cuda:0, cpu, or auto")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument("--steps", type=int, default=10, help="Denoising steps")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional rollout step limit; defaults to the LIBERO suite horizon.",
    )
    parser.add_argument("--domain_id", type=int, default=3, help="XVLA domain id for LIBERO")
    parser.add_argument("--act_type", default="abs", choices=["abs", "rel"], help="LIBERO action type")
    parser.add_argument("--output_dir", default="logs_direct/", help="Directory for logs and videos")
    parser.add_argument("--no_video", action="store_true", help="Skip mp4 video writing")
    parser.print_help()


if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    _print_help_without_runtime_deps()
    raise SystemExit(0)

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Allow running as:
#   python evaluation/libero/libero_direct_eval.py
# from the X-VLA repository root, or directly from this folder.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import robosuite.utils.transform_utils as T

from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor


EPS = 1e-6

LIBERO_DATASETS = {
    "libero_goal": ["libero_goal"],
    "libero_object": ["libero_object"],
    "libero_spatial": ["libero_spatial"],
    "libero_10": ["libero_10"],
    "libero_90": ["libero_90"],
    "libero30": ["libero_goal", "libero_object", "libero_spatial"],
    "libero130": ["libero_goal", "libero_object", "libero_spatial", "libero_10", "libero_90"],
}

LIBERO_DATASETS_HORIZON = {
    "libero_goal": 800,
    "libero_object": 800,
    "libero_spatial": 800,
    "libero_10": 900,
    "libero_90": 800,
    "libero30": 800,
    "libero130": 800,
}

DEFAULT_TASK_SUITES = ["libero_10", "libero_spatial", "libero_goal", "libero_object"]
DEFAULT_SUITE_ADAPTER_FOLDERS = {
    "libero_goal": "goal",
    "libero_object": "object",
    "libero_10": "long",
    "libero_spatial": "spatial",
}
benchmark_dict = benchmark.get_benchmark_dict()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _flip_agentview(img: np.ndarray) -> np.ndarray:
    """Match the original LIBERO client behavior."""
    return np.flip(np.flip(img, 0), 1)


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "rollout"


def _to_pil_image(array: np.ndarray) -> Image.Image:
    array = np.ascontiguousarray(array)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def torch_dtype_from_name(name: str) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[name]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rollout_seed(base_seed: int, task_id: int, ep: int) -> int:
    return base_seed + task_id * 10000 + ep


def parse_task_suites(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return list(DEFAULT_TASK_SUITES)

    suites: List[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                suites.append(item)

    invalid = [suite for suite in suites if suite not in LIBERO_DATASETS]
    if invalid:
        raise ValueError(
            f"Unknown task suite(s): {invalid}. Available: {sorted(LIBERO_DATASETS)}"
        )
    return suites


def parse_task_ids(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None

    text = value.strip()
    if not text or text.lower() == "all":
        return None

    task_ids: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending task id range: {part}")
            task_ids.extend(range(start, end + 1))
        else:
            task_ids.append(int(part))

    if any(task_id < 0 for task_id in task_ids):
        raise ValueError(f"Task ids must be non-negative: {task_ids}")

    deduped: List[int] = []
    seen = set()
    for task_id in task_ids:
        if task_id not in seen:
            seen.add(task_id)
            deduped.append(task_id)
    return deduped


def parse_suite_adapter_map(value: Optional[str]) -> Dict[str, str]:
    mapping = dict(DEFAULT_SUITE_ADAPTER_FOLDERS)
    if value is None or not value.strip():
        return mapping

    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "Suite adapter map entries must look like suite=folder, "
                f"got: {item}"
            )
        suite, folder = item.split("=", 1)
        suite = suite.strip()
        folder = folder.strip().strip("/")
        if suite not in LIBERO_DATASETS:
            raise ValueError(f"Unknown LIBERO suite in adapter map: {suite}")
        if not folder:
            raise ValueError(f"Empty adapter folder for suite: {suite}")
        mapping[suite] = folder
    return mapping


def append_checkpoint_subdir(folder: str, checkpoint_subdir: Optional[str]) -> str:
    if checkpoint_subdir is None or not checkpoint_subdir.strip():
        return folder.strip("/")
    return f"{folder.strip('/')}/{checkpoint_subdir.strip().strip('/')}"


class LiberoAbsActionProcessor:
    """Helpers to convert between 6D rotation and axis-angle."""

    def Rotate6D_to_AxisAngle(self, r6d: np.ndarray) -> np.ndarray:
        single = False
        if r6d.ndim == 1:
            r6d = r6d[None, :]
            single = True

        a1 = r6d[:, 0:3]
        a2 = r6d[:, 3:6]

        b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + EPS)
        dot_prod = np.sum(b1 * a2, axis=-1, keepdims=True)
        b2_orth = a2 - dot_prod * b1
        b2 = b2_orth / (np.linalg.norm(b2_orth, axis=-1, keepdims=True) + EPS)
        b3 = np.cross(b1, b2, axis=-1)

        rot_mats = np.stack([b1, b2, b3], axis=-1)
        axis_angle = []
        for i in range(rot_mats.shape[0]):
            quat = T.mat2quat(rot_mats[i])
            axis_angle.append(T.quat2axisangle(quat))

        result = np.stack(axis_angle, axis=0)
        return result[0] if single else result

    def Mat_to_Rotate6D(self, rot_mat: np.ndarray) -> np.ndarray:
        if rot_mat.ndim == 2:
            return np.concatenate([rot_mat[:3, 0], rot_mat[:3, 1]], axis=-1)
        if rot_mat.ndim == 3:
            return np.concatenate([rot_mat[:, :3, 0], rot_mat[:, :3, 1]], axis=-1)
        raise ValueError("Rotation matrix must be (..., 3, 3)")


class DirectXVLAAgent:
    """In-process XVLA policy wrapper for LIBERO rollouts."""

    def __init__(
        self,
        model_path: str,
        processor_path: Optional[str] = None,
        lora_path: Optional[str] = None,
        adapter_mode: str = "single",
        suite_lora_root: Optional[str] = None,
        suite_lora_map: Optional[str] = None,
        suite_lora_checkpoint_subdir: Optional[str] = None,
        fuse_first_lora_path: Optional[str] = None,
        fuse_rest_lora_path: Optional[str] = None,
        fuse_switch_step: int = 20,
        device: str = "cuda",
        dtype: str = "float32",
        steps: int = 10,
        domain_id: int = 3,
    ) -> None:
        self.device = self._resolve_device(device)
        self.torch_dtype = torch_dtype_from_name(dtype)
        self.steps = steps
        self.domain_id = domain_id
        self.processor_utils = LiberoAbsActionProcessor()
        self.adapter_mode = adapter_mode
        self.fuse_switch_step = fuse_switch_step
        self.suite_lora_root = suite_lora_root
        self.suite_lora_map = parse_suite_adapter_map(suite_lora_map)
        self.suite_lora_checkpoint_subdir = suite_lora_checkpoint_subdir
        self.loaded_adapter_names: set[str] = set()
        self.active_adapter_name: Optional[str] = None

        processor_path = processor_path or model_path
        print(f"Loading XVLAProcessor from: {processor_path}")
        self.processor = XVLAProcessor.from_pretrained(processor_path)

        print(f"Loading XVLA model from: {model_path}")
        self.model = XVLA.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
        )

        if adapter_mode == "fuse":
            first_path = fuse_first_lora_path or lora_path
            if not first_path:
                raise ValueError("fuse mode requires --fuse_first_lora_path or --lora_path")
            if not fuse_rest_lora_path:
                raise ValueError("fuse mode requires --fuse_rest_lora_path")

            print(f"Loading fuse first adapter from: {first_path}")
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model,
                first_path,
                adapter_name="fuse_first",
                torch_dtype=self.torch_dtype,
            )
            self.loaded_adapter_names.add("fuse_first")
            print(f"Loading fuse rest adapter from: {fuse_rest_lora_path}")
            self.model.load_adapter(fuse_rest_lora_path, adapter_name="fuse_rest")
            self.loaded_adapter_names.add("fuse_rest")
            self.active_adapter_name = "fuse_first"
        elif lora_path:
            print(f"Applying LoRA weights from: {lora_path}")
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model,
                lora_path,
                adapter_name="single",
                torch_dtype=self.torch_dtype,
            )
            self.loaded_adapter_names.add("single")
            self.active_adapter_name = "single"
        elif adapter_mode == "suite":
            if not suite_lora_root:
                raise ValueError("suite mode requires --suite_lora_root")
        elif adapter_mode != "single":
            raise ValueError(f"Unsupported adapter_mode: {adapter_mode}")

        self.model = self.model.to(self.device).to(self.torch_dtype)
        self.model.eval()
        self.reset()

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device(device)

    def _initial_adapter_name(self) -> Optional[str]:
        if self.adapter_mode == "fuse":
            return "fuse_first"
        if self.adapter_mode == "suite":
            return self.active_adapter_name
        return "single" if self.active_adapter_name == "single" else None

    def _adapter_name_for_step(self, env_step: int) -> Optional[str]:
        if self.adapter_mode == "fuse":
            return "fuse_first" if env_step < self.fuse_switch_step else "fuse_rest"
        if self.adapter_mode == "suite":
            return self.active_adapter_name
        return "single" if self.active_adapter_name == "single" else None

    def _set_active_adapter(self, adapter_name: Optional[str]) -> None:
        if adapter_name is None:
            return
        if self.active_adapter_name == adapter_name:
            return
        if not hasattr(self.model, "set_adapter"):
            raise RuntimeError("Adapter switching requested, but model does not support set_adapter().")
        print(f"Switching active adapter to: {adapter_name}")
        self.model.set_adapter(adapter_name)
        self.active_adapter_name = adapter_name

    def _prepare_adapter_for_step(self, env_step: int) -> None:
        adapter_name = self._adapter_name_for_step(env_step)
        if adapter_name != self.active_adapter_name:
            self.action_plan.clear()
            self._set_active_adapter(adapter_name)

    def _adapter_source_for_suite(self, suite_name: str) -> tuple[str, Dict[str, str]]:
        if suite_name not in self.suite_lora_map:
            raise ValueError(
                f"No suite adapter folder is configured for {suite_name}. "
                "Use concrete suites like libero_goal/libero_object/libero_10/libero_spatial, "
                "or pass --suite_lora_map."
            )

        folder = append_checkpoint_subdir(
            self.suite_lora_map[suite_name],
            self.suite_lora_checkpoint_subdir,
        )
        root_path = Path(self.suite_lora_root).expanduser() if self.suite_lora_root else None
        if root_path is not None and root_path.exists():
            adapter_path = root_path / Path(folder)
            if not adapter_path.exists():
                raise FileNotFoundError(f"Suite adapter folder does not exist: {adapter_path}")
            return str(adapter_path), {}
        return str(self.suite_lora_root), {"subfolder": folder}

    def set_suite_adapter(self, suite_name: str) -> None:
        if self.adapter_mode != "suite":
            return

        adapter_name = f"suite_{suite_name}"
        if adapter_name not in self.loaded_adapter_names:
            source, kwargs = self._adapter_source_for_suite(suite_name)
            print(f"Loading adapter for {suite_name}: source={source}, options={kwargs}")
            if not self.loaded_adapter_names:
                from peft import PeftModel

                self.model = PeftModel.from_pretrained(
                    self.model,
                    source,
                    adapter_name=adapter_name,
                    torch_dtype=self.torch_dtype,
                    **kwargs,
                )
                self.model = self.model.to(self.device).to(self.torch_dtype)
                self.model.eval()
            else:
                if not hasattr(self.model, "load_adapter"):
                    raise RuntimeError("Suite adapter mode requires a PEFT model with load_adapter().")
                self.model.load_adapter(source, adapter_name=adapter_name, **kwargs)
            self.loaded_adapter_names.add(adapter_name)

        self.action_plan.clear()
        self._set_active_adapter(adapter_name)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            seed_everything(seed)
        self.proprio: Optional[np.ndarray] = None
        self.action_plan: Deque[List[float]] = collections.deque()
        self._set_active_adapter(self._initial_adapter_name())

    def _to_model(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_floating_point():
            return tensor.to(device=self.device, dtype=self.torch_dtype)
        return tensor.to(device=self.device)

    def _format_inputs(self, obs: Dict, goal: str) -> Dict[str, torch.Tensor]:
        main_view = _flip_agentview(obs["agentview_image"])
        wrist_view = obs["robot0_eye_in_hand_image"]
        images = [_to_pil_image(main_view), _to_pil_image(wrist_view)]

        closed_loop_proprio = np.concatenate(
            [obs["robo_pos"], obs["robo_ori"], np.array([0.0], dtype=np.float32)],
            axis=-1,
        ).astype(np.float32)
        closed_loop_proprio = np.concatenate(
            [closed_loop_proprio, np.zeros_like(closed_loop_proprio)],
            axis=-1,
        )
        if self.proprio is None:
            self.proprio = closed_loop_proprio

        inputs = self.processor(images, goal)
        required = {"input_ids", "image_input", "image_mask"}
        if not required.issubset(inputs):
            missing = sorted(required - set(inputs))
            raise RuntimeError(f"Processor returned incomplete inputs; missing: {missing}")

        model_inputs = {key: self._to_model(value) for key, value in inputs.items()}
        model_inputs["proprio"] = self._to_model(torch.as_tensor(self.proprio).unsqueeze(0))
        model_inputs["domain_id"] = torch.tensor(
            [self.domain_id],
            dtype=torch.long,
            device=self.device,
        )
        return model_inputs

    def _infer_action_plan(self, obs: Dict, goal: str) -> np.ndarray:
        model_inputs = self._format_inputs(obs, goal)
        action = self.model.generate_actions(**model_inputs, steps=self.steps)
        action = action.squeeze(0).float().cpu().numpy()
        if action.ndim != 2 or action.shape[1] < 10:
            raise RuntimeError(f"Unexpected action shape from model: {action.shape}")
        return action

    def step(self, obs: Dict, goal: str, env_step: int = 0) -> np.ndarray:
        self._prepare_adapter_for_step(env_step)
        if not self.action_plan:
            action = self._infer_action_plan(obs, goal)

            assert self.proprio is not None
            self.proprio[:9] = action[-1, :9].copy()

            target_eef = action[:, :3]
            target_axis = self.processor_utils.Rotate6D_to_AxisAngle(action[:, 3:9])
            target_gripper = action[:, 9:10]
            final_action = np.concatenate([target_eef, target_axis, target_gripper], axis=-1)

            for row in final_action.tolist():
                self.action_plan.append(row)

        action_predict = np.array(self.action_plan.popleft(), dtype=np.float32)
        action_predict[-1] = 1.0 if action_predict[-1] > 0.5 else -1.0
        return action_predict


class DirectLIBEROEval:
    def __init__(
        self,
        task_suite_name: str,
        task_ids: Optional[Sequence[int]] = None,
        eval_horizon: int = 600,
        act_type: str = "abs",
        num_episodes: int = 10,
        env_seed: int = 42,
        policy_seed: int = 42,
        no_video: bool = False,
    ) -> None:
        self.task_suite_name = task_suite_name
        self.task_list = LIBERO_DATASETS[self.task_suite_name]
        self.task_suite_list = [benchmark_dict[task]() for task in self.task_list]
        self.task_ids = list(task_ids) if task_ids is not None else None
        self.eval_horizon = eval_horizon
        self.num_episodes = num_episodes
        self.env_seed = env_seed
        self.policy_seed = policy_seed
        self.act_type = act_type
        self.no_video = no_video
        self.processor = LiberoAbsActionProcessor()
        self.base_dir: Path = Path(".")

    def _make_dir(self, save_path: Path) -> None:
        path = save_path / self.task_suite_name
        _ensure_dir(path)
        self.base_dir = path

    def _selected_task_ids(self, task_suite) -> List[int]:
        available = len(task_suite.tasks)
        if self.task_ids is None:
            return list(range(available))

        invalid = [task_id for task_id in self.task_ids if task_id >= available]
        if invalid:
            raise ValueError(
                f"Task id(s) {invalid} are out of range for {self.task_suite_name}; "
                f"available ids are 0..{available - 1}"
            )
        return list(self.task_ids)

    def _init_env(self, task_suite, task_id: int, ep: int):
        task = task_suite.get_task(task_id)
        task_description = task.language
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file,
        )
        print(
            f"[info] retrieving task {task_id} from suite {self.task_suite_name}, "
            f"language: {task_description}, bddl: {task_bddl_file}"
        )

        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": 256,
            "camera_widths": 256,
        }
        env = OffScreenRenderEnv(**env_args)

        env.seed(rollout_seed(self.env_seed, task_id, ep) + 100)
        obs = env.reset()
        init_states = task_suite.get_task_init_states(task_id)
        init_state_id = ep % init_states.shape[0]
        obs = env.set_init_state(init_states[init_state_id])

        for _ in range(10):
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            obs, reward, done, info = env.step(action)

        if self.act_type == "abs":
            for robot in env.env.robots:
                robot.controller.use_delta = False
        elif self.act_type != "rel":
            raise ValueError("act_type must be 'abs' or 'rel'")

        return env, task_description, obs

    def _log_results(self, metrics: Dict) -> None:
        print(metrics)
        save_name = self.base_dir / "rollout_results.jsonl"
        with save_name.open("a+", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")

    def _save_video(self, save_path: Path, images: List[np.ndarray], fps: int = 30) -> None:
        import imageio

        imageio.mimsave(save_path.as_posix(), images, fps=fps)

    def _rollout(self, task_suite, policy: DirectXVLAAgent, task_id: int, ep: int) -> float:
        env, lang, obs = self._init_env(task_suite, task_id, ep)
        seed_everything(rollout_seed(self.policy_seed, task_id, ep))
        images: List[np.ndarray] = []

        done_flag = False
        try:
            for step_idx in tqdm(range(self.eval_horizon), desc=f"{lang}"):
                robo_ori = self.processor.Mat_to_Rotate6D(env.env.robots[0].controller.ee_ori_mat)
                robo_pos = env.env.robots[0].controller.ee_pos
                obs["robo_ori"] = robo_ori
                obs["robo_pos"] = robo_pos

                action = policy.step(obs, lang, env_step=step_idx)

                if not self.no_video:
                    images.append(_flip_agentview(obs["agentview_image"]))
                obs, reward, done, info = env.step(action)
                if done:
                    done_flag = True
                    break
        finally:
            env.close()

        if not self.no_video and images:
            video_name = f"{_safe_filename(lang)}_{task_id}_{ep}.mp4"
            self._save_video(self.base_dir / video_name, images, fps=30)

        success = 1.0 if done_flag else 0.0
        metrics = {
            f"sim/{self.task_suite_name}/{lang}": success,
            "task_id": task_id,
            "episode": ep,
        }
        self._log_results(metrics)
        return success

    def eval_episodes(self, policy: DirectXVLAAgent, save_path: Path) -> float:
        self._make_dir(save_path)

        rewards: List[float] = []
        for task_suite in self.task_suite_list:
            selected_ids = self._selected_task_ids(task_suite)
            for task_id in tqdm(selected_ids, desc="Evaluating tasks"):
                for ep in range(self.num_episodes):
                    policy.reset(seed=rollout_seed(self.policy_seed, task_id, ep))
                    rewards.append(self._rollout(task_suite, policy, task_id, ep))

        eval_rewards = float(sum(rewards) / max(len(rewards), 1))
        self._log_results({f"sim_summary/{self.task_suite_name}/all": eval_rewards})
        return eval_rewards


def eval_libero_direct(
    agent: DirectXVLAAgent,
    save_path: Path,
    task_suites: Iterable[str],
    task_ids: Optional[Sequence[int]] = None,
    num_episodes: int = 10,
    env_seed: int = 42,
    policy_seed: int = 42,
    act_type: str = "abs",
    no_video: bool = False,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    result_dict: Dict[str, float] = {}
    _ensure_dir(save_path)

    for suite_name in task_suites:
        agent.set_suite_adapter(suite_name)
        horizon = max_steps if max_steps is not None else LIBERO_DATASETS_HORIZON[suite_name]
        evaluator = DirectLIBEROEval(
            task_suite_name=suite_name,
            task_ids=task_ids,
            eval_horizon=horizon,
            act_type=act_type,
            num_episodes=num_episodes,
            env_seed=env_seed,
            policy_seed=policy_seed,
            no_video=no_video,
        )
        result_dict[suite_name] = evaluator.eval_episodes(agent, save_path=save_path)

    with (save_path / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result_dict, handle, indent=2)
        handle.write("\n")
    return result_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Direct LIBERO evaluation for X-VLA")
    parser.add_argument("--model_path", required=True, help="XVLA checkpoint path or Hugging Face id")
    parser.add_argument(
        "--processor_path",
        default=None,
        help="Processor path or id; defaults to --model_path",
    )
    parser.add_argument("--lora_path", default=None, help="Optional LoRA adapter path")
    parser.add_argument(
        "--adapter_mode",
        default="single",
        choices=["single", "fuse", "suite"],
        help="Adapter routing mode. 'suite' chooses adapter folder from task suite.",
    )
    parser.add_argument(
        "--suite_lora_root",
        default=None,
        help="HF repo id or local root containing goal/object/long/spatial adapter folders.",
    )
    parser.add_argument(
        "--suite_lora_map",
        default=None,
        help="Optional suite=folder overrides, e.g. libero_10=long,libero_goal=goal.",
    )
    parser.add_argument(
        "--suite_lora_checkpoint_subdir",
        default=None,
        help="Optional checkpoint subfolder appended inside each suite folder.",
    )
    parser.add_argument(
        "--fuse_first_lora_path",
        default=None,
        help="First adapter for fuse mode; defaults to --lora_path.",
    )
    parser.add_argument(
        "--fuse_rest_lora_path",
        default=None,
        help="Adapter used after --fuse_switch_step in fuse mode.",
    )
    parser.add_argument(
        "--fuse_switch_step",
        type=int,
        default=20,
        help="LIBERO env step where fuse mode switches to the rest adapter.",
    )
    parser.add_argument(
        "--task_suites",
        nargs="+",
        default=DEFAULT_TASK_SUITES,
        help="LIBERO suite names; accepts space-separated or comma-separated values",
    )
    parser.add_argument(
        "--task_ids",
        default=None,
        help="Task ids to evaluate, e.g. 0,1,4-7. Defaults to all tasks.",
    )
    parser.add_argument("--eval_time", type=int, default=10, help="Episodes per selected task")
    parser.add_argument("--init_seed", type=int, default=42, help="Fallback seed for env and policy")
    parser.add_argument(
        "--env_seed",
        type=int,
        default=None,
        help="LIBERO environment seed; defaults to --init_seed.",
    )
    parser.add_argument(
        "--policy_seed",
        type=int,
        default=None,
        help="Python/NumPy/Torch/CUDA policy seed; defaults to --init_seed.",
    )
    parser.add_argument("--device", default="cuda", help="Device: cuda, cuda:0, cpu, or auto")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument("--steps", type=int, default=10, help="Denoising steps")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional rollout step limit; defaults to the LIBERO suite horizon.",
    )
    parser.add_argument("--domain_id", type=int, default=3, help="XVLA domain id for LIBERO")
    parser.add_argument("--act_type", default="abs", choices=["abs", "rel"], help="LIBERO action type")
    parser.add_argument("--output_dir", default="logs_direct/", help="Directory for logs and videos")
    parser.add_argument("--no_video", action="store_true", help="Skip mp4 video writing")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        task_suites = parse_task_suites(args.task_suites)
        task_ids = parse_task_ids(args.task_ids)
    except ValueError as exc:
        parser.error(str(exc))

    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max_steps must be greater than 0")
    if args.fuse_switch_step <= 0:
        parser.error("--fuse_switch_step must be greater than 0")
    if args.adapter_mode == "fuse":
        if not (args.fuse_first_lora_path or args.lora_path):
            parser.error("fuse mode requires --fuse_first_lora_path or --lora_path")
        if not args.fuse_rest_lora_path:
            parser.error("fuse mode requires --fuse_rest_lora_path")
    if args.adapter_mode == "suite":
        if not args.suite_lora_root:
            parser.error("suite mode requires --suite_lora_root")
        try:
            suite_lora_map = parse_suite_adapter_map(args.suite_lora_map)
        except ValueError as exc:
            parser.error(str(exc))
        missing_suites = [suite for suite in task_suites if suite not in suite_lora_map]
        if missing_suites:
            parser.error(
                "suite mode has no adapter folder for: "
                f"{missing_suites}. Pass --suite_lora_map to add them."
            )

    env_seed = args.env_seed if args.env_seed is not None else args.init_seed
    policy_seed = args.policy_seed if args.policy_seed is not None else args.init_seed

    output_dir = Path(args.output_dir)
    _ensure_dir(output_dir)

    print("Starting direct LIBERO evaluation.")
    print(f"model_path: {args.model_path}")
    print(f"processor_path: {args.processor_path or args.model_path}")
    print(f"lora_path: {args.lora_path}")
    print(f"adapter_mode: {args.adapter_mode}")
    print(f"suite_lora_root: {args.suite_lora_root}")
    print(f"suite_lora_map: {args.suite_lora_map or DEFAULT_SUITE_ADAPTER_FOLDERS}")
    print(f"suite_lora_checkpoint_subdir: {args.suite_lora_checkpoint_subdir}")
    print(f"fuse_first_lora_path: {args.fuse_first_lora_path or args.lora_path}")
    print(f"fuse_rest_lora_path: {args.fuse_rest_lora_path}")
    print(f"fuse_switch_step: {args.fuse_switch_step}")
    print(f"task_suites: {task_suites}")
    print(f"task_ids: {task_ids if task_ids is not None else 'all'}")
    print(f"episodes per task: {args.eval_time}")
    print(f"env_seed: {env_seed}")
    print(f"policy_seed: {policy_seed}")
    print(f"device: {args.device}")
    print(f"dtype: {args.dtype}")
    print(f"steps: {args.steps}")
    print(f"max_steps: {args.max_steps if args.max_steps is not None else 'suite default'}")
    print(f"domain_id: {args.domain_id}")
    print(f"act_type: {args.act_type}")
    print(f"output_dir: {output_dir.resolve()}")
    print(f"no_video: {args.no_video}")

    try:
        agent = DirectXVLAAgent(
            model_path=args.model_path,
            processor_path=args.processor_path,
            lora_path=args.lora_path,
            adapter_mode=args.adapter_mode,
            suite_lora_root=args.suite_lora_root,
            suite_lora_map=args.suite_lora_map,
            suite_lora_checkpoint_subdir=args.suite_lora_checkpoint_subdir,
            fuse_first_lora_path=args.fuse_first_lora_path,
            fuse_rest_lora_path=args.fuse_rest_lora_path,
            fuse_switch_step=args.fuse_switch_step,
            device=args.device,
            dtype=args.dtype,
            steps=args.steps,
            domain_id=args.domain_id,
        )
        results = eval_libero_direct(
            agent=agent,
            save_path=output_dir,
            task_suites=task_suites,
            task_ids=task_ids,
            num_episodes=args.eval_time,
            env_seed=env_seed,
            policy_seed=policy_seed,
            act_type=args.act_type,
            no_video=args.no_video,
            max_steps=args.max_steps,
        )
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 130
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2

    print("Direct LIBERO evaluation completed.")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
