import logging
import time
import itertools
from pathlib import Path
from typing import List, Union

import ray
import torch

from slime.rollout.base_types import call_rollout_fn
from slime.utils.health_monitor import RolloutHealthMonitor
from slime.utils.http_utils import init_http_client
from slime.utils.logging_utils import configure_logger
from slime.utils.metric_utils import MetricChecker
from slime.utils.misc import load_function
from slime.utils.misc import Box
from slime.utils.logging_utils import init_tracking
from slime.utils.types import Sample

from slime.ray.utils import Lock
from slime.ray.rollout import _start_router, init_rollout_engines, _log_rollout_data, _log_eval_rollout_data

from rollout_buffer import GymRolloutDataSource
from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions
from slime.utils import logging_utils

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

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

        # data_source_cls = load_function(self.args.data_source_path)
        self.data_source = GymRolloutDataSource(args)

        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.dynamic_dataset_rollout = load_function(self.args.dynamic_datarollout_path)
        self.custom_reward_post_process_func = None
        if self.args.custom_reward_post_process_path is not None:
            self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
        self.custom_convert_samples_to_train_data_func = None
        if self.args.custom_convert_samples_to_train_data_path is not None:
            self.custom_convert_samples_to_train_data_func = load_function(
                self.args.custom_convert_samples_to_train_data_path
            )
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
        self.rollout_id = -1

        self._metric_checker = MetricChecker.maybe_create(args)
        self._health_monitor = None
        if self.args.use_fault_tolerance:
            self._health_monitor = RolloutHealthMonitor(self, args)
            self._health_monitor.start()  # Start the monitor thread (in paused state)
            self._ci_fault_injection_pending = self.args.ci_test  # Flag for CI fault injection

    def _try_ci_fault_injection(self):
        """Try to inject fault during generate (when health monitor is running)."""
        if not self._ci_fault_injection_pending:
            return

        # Only inject fault once
        self._ci_fault_injection_pending = False

        if self.all_rollout_engines and self.all_rollout_engines[0]:
            logger.info("CI Fault Injection: Simulating crash on engine 0 during generate")
            try:
                # This will cause the ray actor to exit
                self.all_rollout_engines[0].simulate_crash.remote()
                # Wait for health monitor to detect the crash and mark engine as None
                # health_check_interval + health_check_timeout + buffer
                wait_time = self.args.rollout_health_check_interval + self.args.rollout_health_check_timeout + 5
                logger.info(f"CI Fault Injection: Waiting {wait_time}s for health monitor to detect crash")
                time.sleep(wait_time)
            except Exception as e:
                logger.warning(f"CI Fault Injection failed: {e}")

    def dispose(self):
        if self._metric_checker is not None:
            self._metric_checker.dispose()
        if self._health_monitor is not None:
            self._health_monitor.stop()

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
        start_time = time.time()
        self.rollout_id = rollout_id
        self.health_monitoring_resume()
        if self.args.ci_test and self.args.use_fault_tolerance and rollout_id >= 2:
            self._try_ci_fault_injection()
        data, metrics = self._get_rollout_data(rollout_id=rollout_id)
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=False)
        _log_rollout_data(rollout_id, self.args, data, metrics, time.time() - start_time)
        data = self._convert_samples_to_train_data(data)
        return self._split_train_data_by_dp(data, self.train_parallel_config["dp_size"])

    def eval(self, rollout_id):
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return
        self.health_monitoring_resume()

        result = call_rollout_fn(self.eval_generate_rollout, self.args, rollout_id, self.data_source, evaluation=True)
        data = result.data
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=True)
        metrics = _log_eval_rollout_data(rollout_id, self.args, data, result.metrics)
        if self._metric_checker is not None:
            self._metric_checker.on_eval(metrics)
            
    def evaluate_dataset(self, rollout_id):
        """
        Inside this function, we mimic the eval function above, 
        but we evaluate the whole dataset
        
        We should add a full_dataset option in self.data_source
        """
        if self.args.debug_train_only:
            return
        
        self.health_monitoring_resume()
        
        result = self.dynamic_dataset_rollout(self.args, rollout_id, self.data_source)
        self.data_source.select_dataset(result, keep_min=1, keep_max=6)
        
        num_solvables = sum([1 for raw in result.values() if raw > 0])
        
        _log_dynamic_dataset_metrics(rollout_id, self.args, num_solvables, len(result.keys()))

    def save(self, rollout_id):
        self.data_source.save(rollout_id)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)

    def offload(self):
        self.health_monitoring_pause()
        return ray.get(
            [engine.release_memory_occupation.remote() for engine in self.rollout_engines if engine is not None]
        )

    def onload(self, tags: list[str] | None = None):
        return ray.get(
            [
                engine.resume_memory_occupation.remote(tags=tags)
                for engine in self.rollout_engines
                if engine is not None
            ]
        )

    def onload_weights(self):
        self.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    def onload_kv(self):
        self.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])

    def recover_rollout_engines(self):
        """Restart any dead rollout engines and update num_new_engines for update_weights detection."""
        self.health_monitoring_pause()
        if self.rollout_id == -1:
            return self.rollout_engines, self.rollout_engine_lock, self.num_new_engines

        dead_indices = [i for i, engine in enumerate(self.all_rollout_engines) if engine is None]
        self.num_new_engines = init_rollout_engines(self.args, self.pg, self.all_rollout_engines)
        logger.info(f"Recovered {self.num_new_engines} dead rollout engines")
        assert self.num_new_engines == len(dead_indices), "num_new_engines does not match dead_indices length"
        if self.args.offload_rollout and dead_indices:
            new_engines = [self.all_rollout_engines[i] for i in dead_indices]
            ray.get([engine.release_memory_occupation.remote() for engine in new_engines])
            ray.get([engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]) for engine in new_engines])

        return self.rollout_engines, self.rollout_engine_lock, self.num_new_engines

    def clear_num_new_engines(self):
        # when fault tolerance is not enabled, we need to manually clear num_new_engines after update_weights
        self.num_new_engines = 0

    def health_monitoring_pause(self) -> None:
        if self._health_monitor is not None:
            self._health_monitor.pause()

    def health_monitoring_resume(self) -> None:
        if self._health_monitor is not None:
            self._health_monitor.resume()

    def check_weights(self, action: str):
        return ray.get([engine.check_weights.remote(action=action) for engine in self.rollout_engines])

    def _get_rollout_data(self, rollout_id):
        if self.args.load_debug_rollout_data:
            data = torch.load(
                self.args.load_debug_rollout_data.format(rollout_id=rollout_id),
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
                data = list(itertools.chain.from_iterable(data))

            if not self.args.disable_rollout_trim_samples:
                global_batch_size = self.args.global_batch_size
                if self.args.use_dynamic_global_batch_size:
                    logger.info(f"Collected {len(data)} samples from rollout to train with dynamic global batch size")
                    # TODO: this is a temporary solution, we should directly save dynamic_global_batch_size to rollout data
                    self._dynamic_global_batch_size = self._compute_dynamic_global_batch_size(len(data))
                    global_batch_size = self._dynamic_global_batch_size

                if len(data) % global_batch_size != 0:
                    trim_len = (len(data) // global_batch_size) * global_batch_size
                    if trim_len == 0:
                        raise ValueError(f"Not enough samples {len(data)} for global_batch_size {global_batch_size}")
                    origin_data_length = len(data)
                    data = data[:trim_len]
                    logger.info(f"trim number of samples from {origin_data_length} to {trim_len}")
                logger.info(f"Final collected {len(data)} samples from rollout to train")

        return data, metrics

    def _compute_dynamic_global_batch_size(self, num_samples: int) -> int:
        """Calculate dynamic global_batch_size to ensure only one training step.

        Strategy: global_batch_size = num_samples rounded down to a multiple of dp_size
        This ensures num_steps_per_rollout = num_samples // global_batch_size = 1
        """
        dp_size = self.train_parallel_config["dp_size"]
        original_gbs = self.args.global_batch_size

        # Round down to a multiple of dp_size to ensure only one training step
        dynamic_gbs = (num_samples // dp_size) * dp_size

        if dynamic_gbs == 0:
            # Too few samples, use at least dp_size
            dynamic_gbs = dp_size
            logger.warning(f"num_samples={num_samples} < dp_size={dp_size}, using dp_size as global_batch_size")

        # Calculate how many samples will be discarded
        wasted = num_samples - dynamic_gbs

        if dynamic_gbs != original_gbs or wasted > 0:
            logger.info(
                f"Dynamic global_batch_size: {original_gbs} -> {dynamic_gbs} "
                f"(num_samples={num_samples}, dp_size={dp_size}, "
                f"num_steps=1, wasted={wasted})"
            )

        return dynamic_gbs

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

    def _post_process_rewards(self, samples: list[Sample] | list[list[Sample]]):
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

    def _convert_samples_to_train_data(self, samples: list[Sample] | list[list[Sample]]):
        """
        Convert inference generated samples to training data.
        """
        if self.custom_convert_samples_to_train_data_func is not None:
            return self.custom_convert_samples_to_train_data_func(self.args, samples)

        raw_rewards, rewards = self._post_process_rewards(samples)

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
            if sample.remove_sample:
                sample.loss_mask = [0] * sample.response_length
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

        if samples[0].multimodal_train_inputs is not None:
            train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

        if "teacher_log_probs" in samples[0].__dict__:
            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

        return train_data

    def set_train_parallel_config(self, config: dict):
        self.train_parallel_config = config

    def _split_train_data_by_dp(self, data, dp_size):
        """Split the train data by data parallel size."""
        rollout_data = {}

        if "prompt" in data:
            rollout_data["prompt"] = data["prompt"]

        total_lengths = [len(t) for t in data["tokens"]]
        data["total_lengths"] = total_lengths

        if self.args.balance_data:
            partitions = get_seqlen_balanced_partitions(total_lengths, dp_size, equal_size=True)
        else:
            partitions = [range(i, len(total_lengths), dp_size) for i in range(dp_size)]

        rollout_data_refs = []

        for i in range(dp_size):
            rollout_data = {}
            partition = partitions[i]
            rollout_data["partition"] = partition
            for key in [
                "tokens",
                "multimodal_train_inputs",
                "response_lengths",
                "rewards",
                "truncated",
                "loss_masks",
                "round_number",
                "sample_indices",
                "rollout_log_probs",
                "rollout_routed_experts",
                "prompt",
                "teacher_log_probs",
            ]:
                if key not in data:
                    continue
                val = [data[key][j] for j in partition]
                rollout_data[key] = val
            # keys that need to be splited at train side
            for key in [
                "raw_reward",
                "total_lengths",
            ]:
                if key not in data:
                    continue
                rollout_data[key] = data[key]
            # Pass dynamic global_batch_size to training side
            if hasattr(self, "_dynamic_global_batch_size"):
                rollout_data["dynamic_global_batch_size"] = self._dynamic_global_batch_size
            rollout_data_refs.append(Box(ray.put(rollout_data)))
        return rollout_data_refs


def _log_dynamic_dataset_metrics(rollout_id, args, num_solvables, total_size):
    log_dict = {"dynamic_dataset/num_solvables": num_solvables}
    solve_rate = num_solvables / total_size
    log_dict["dynamic_dataset/solve_rate"] = solve_rate
    
    log_dict["dynamic_dataset/step"] = rollout_id
    
    logging_utils.log(args, log_dict, step_key="dynamic_dataset/step")
