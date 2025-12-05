import logging
import time
from pathlib import Path
from typing import List, Union

import ray
import torch

from slime.rollout.base_types import call_rollout_fn
from slime.utils.health_monitor import RolloutHealthMonitor
from slime.utils.http_utils import init_http_client
from slime.utils.logging_utils import configure_logger
from slime.utils.metric_checker import MetricChecker
from slime.utils.misc import load_function
from slime.utils.ray_utils import Box
from slime.utils.tracking_utils import init_tracking
from slime.utils.types import Sample

from slime.ray.utils import Lock
from slime.ray.rollout import _start_router, init_rollout_engines, _log_rollout_data, _log_eval_rollout_data

from rollout_buffer import GymRolloutDataSource

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, pg):
        configure_logger()

        self.args = args
        self.pg = pg
        _start_router(args)
        # TODO make args immutable
        init_tracking(args, primary=False, router_addr=f"http://{args.sglang_router_ip}:{args.sglang_router_port}")
        init_http_client(args)

        self.data_source = GymRolloutDataSource(args)

        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.custom_reward_post_process_func = None
        if self.args.custom_reward_post_process_path is not None:
            self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
        logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
        logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        if self.args.debug_train_only:
            self.all_rollout_engines = []
        else:
            num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, args.num_gpus_per_node)
            num_engines = args.rollout_num_gpus // num_gpu_per_engine
            self.all_rollout_engines = [None] * num_engines
        self.num_new_engines = init_rollout_engines(args, pg, self.all_rollout_engines)
        self.nodes_per_engine = max(1, args.rollout_num_gpus_per_engine // args.num_gpus_per_node)
        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()

        self._metric_checker = MetricChecker.maybe_create(args)
        if self.args.use_fault_tolerance:
            self._health_monitor = RolloutHealthMonitor(self, args)

    def dispose(self):
        if self._metric_checker is not None:
            self._metric_checker.dispose()

    # TODO maybe rename "rollout_engines" and "all_rollout_engines" later
    @property
    def rollout_engines(self):
        # when doing multi-node serving, we will only send request to node-0 for each engine.
        return self.all_rollout_engines[:: self.nodes_per_engine]

    def get_rollout_engines_and_lock(self):
        return self.rollout_engines, self.rollout_engine_lock, self.num_new_engines

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return len(self.data_source.dataset) // self.args.rollout_batch_size

    def generate(self, rollout_id):
        monitor_started = self.args.use_fault_tolerance and self._health_monitor.start()
        start_time = time.time()
        try:
            data, metrics = self._get_rollout_data(rollout_id=rollout_id)
            self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=False)
            _log_rollout_data(rollout_id, self.args, data, metrics, time.time() - start_time)
            data = self._convert_samples_to_train_data(data)
            return Box(ray.put(data))
        finally:
            if monitor_started:
                self._health_monitor.stop()
                self.num_new_engines = init_rollout_engines(self.args, self.pg, self.all_rollout_engines)
            else:
                self.num_new_engines = 0

    def eval(self, rollout_id):
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return
        # TODO: add fault tolerance to eval
        data = call_rollout_fn(
            self.eval_generate_rollout, self.args, rollout_id, self.data_source, evaluation=True
        ).data
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=True)
        metrics = _log_eval_rollout_data(rollout_id, self.args, data)
        if self._metric_checker is not None:
            self._metric_checker.on_eval(metrics)

    def save(self, rollout_id):
        self.data_source.save(rollout_id)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)

    def offload(self):
        return ray.get([engine.release_memory_occupation.remote() for engine in self.rollout_engines])

    def onload(self, tags: List[str] = None):
        return ray.get([engine.resume_memory_occupation.remote(tags=tags) for engine in self.rollout_engines])

    def check_weights(self, action: str):
        return ray.get([engine.check_weights.remote(action=action) for engine in self.rollout_engines])

    def _get_rollout_data(self, rollout_id):
        if self.args.load_debug_rollout_data:
            data = torch.load(
                open(self.args.load_debug_rollout_data.format(rollout_id=rollout_id), "rb"),
                weights_only=False,
            )["samples"]
            data = [Sample.from_dict(sample) for sample in data]
            if (ratio := self.args.load_debug_rollout_data_subsample) is not None:
                original_num_rows = len(data)
                rough_subsample_num_rows = int(original_num_rows * ratio)
                data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
                logger.info(
                    f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}"
                )
            metrics = None
        else:
            data = call_rollout_fn(self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False)
            metrics = data.metrics
            data = data.samples
            # flatten the data if it is a list of lists
            while isinstance(data[0], list):
                data = sum(data, [])
                
            # Lynx: Add warning if failed to get enough data
            if len(data) != self.args.global_batch_size:
                logger.warning(f"Warning: expected {self.args.global_batch_size} samples, got {len(data)}")
                
            # Lynx: Maybe padding? this is really unsafe
            if len(data) % self.args.global_batch_size != 0:
                trim_len = (len(data) // self.args.global_batch_size) * self.args.global_batch_size
                origin_data_length = len(data)
                data = data[:trim_len]
                logger.info(f"trim number of samples from {origin_data_length} to {trim_len}")
        return data, metrics

    def _save_debug_rollout_data(self, data, rollout_id, evaluation: bool):
        # TODO to be refactored (originally Buffer._set_data)
        if (path_template := self.args.save_debug_rollout_data) is not None:
            path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
            logger.info(f"Save debug rollout data to {path}")
            path.parent.mkdir(parents=True, exist_ok=True)

            # TODO may improve the format
            if evaluation:
                dump_data = dict(
                    samples=[sample.to_dict() for dataset_name, info in data.items() for sample in info["samples"]]
                )
            else:
                dump_data = dict(
                    samples=[sample.to_dict() for sample in data],
                )

            torch.save(dict(rollout_id=rollout_id, **dump_data), path)

    def _post_process_rewards(self, samples: Union[list[Sample], list[list[Sample]]]):
        if self.custom_reward_post_process_func is not None:
            return self.custom_reward_post_process_func(self.args, samples)

        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
        if (
            self.args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
            and self.args.rewards_normalization
        ):
            # group norm
            rewards = torch.tensor(raw_rewards, dtype=torch.float)
            if rewards.shape[-1] == self.args.n_samples_per_prompt * self.args.rollout_batch_size:
                rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
            else:
                # when samples count are not equal in each group
                rewards = rewards.view(-1, rewards.shape[-1])
            mean = rewards.mean(dim=-1, keepdim=True)
            rewards = rewards - mean

            if self.args.advantage_estimator in ["grpo", "gspo"] and self.args.grpo_std_normalization:
                std = rewards.std(dim=-1, keepdim=True)
                rewards = rewards / (std + 1e-6)

            return raw_rewards, rewards.flatten().tolist()

        return raw_rewards, raw_rewards

    def _convert_samples_to_train_data(self, samples: Union[list[Sample], list[list[Sample]]]):
        """
        Convert inference generated samples to training data.
        """
        # raw_rewards, rewards = self._post_process_rewards(samples)
        raw_rewards = [sample.reward for sample in samples]
        rewards = [sample.advantage for sample in samples]

        assert len(raw_rewards) == len(samples)
        assert len(rewards) == len(samples)

        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            # some reward model, e.g. remote rm, may return multiple rewards,
            # we could use key to select the reward.
            "rewards": rewards,
            "raw_reward": raw_rewards,
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
        }

        # loss mask
        # TODO: compress the loss mask
        loss_masks = []
        for sample in samples:
            # always instantiate loss_mask if not provided
            if sample.loss_mask is None:
                sample.loss_mask = [1] * sample.response_length

            assert (
                len(sample.loss_mask) == sample.response_length
            ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
            loss_masks.append(sample.loss_mask)
        train_data["loss_masks"] = loss_masks

        # overwriting the raw reward
        if samples[0].metadata and "raw_reward" in samples[0].metadata:
            train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

        # For rollout buffer
        if samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

        # Add rollout log probabilities for off-policy correction
        if samples[0].rollout_log_probs is not None:
            train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

        if samples[0].rollout_routed_experts is not None:
            train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

        if samples[0].train_metadata is not None:
            train_data["metadata"] = [sample.train_metadata for sample in samples]

        if "teacher_log_probs" in samples[0].__dict__:
            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

        return train_data
