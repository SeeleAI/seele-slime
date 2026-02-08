import abc
import copy
import logging
import os
from pathlib import Path

import torch

from slime.utils.data import Dataset
from slime.utils.misc import load_function
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class DataSource(abc.ABC):
    @abc.abstractmethod
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

    @abc.abstractmethod
    def add_samples(self, samples: list[list[Sample]]):
        """
        Add samples to the data source
        """

    @abc.abstractmethod
    def save(self, rollout_id):
        """
        Save the state of the data source
        """

    @abc.abstractmethod
    def load(self, rollout_id=None):
        """
        Load the state of the data source
        """


# TODO may further refactor data-loading part later
class RolloutDataSource(DataSource):
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        # TODO remove this
        self.metadata = {}

        if args.rollout_global_dataset:
            tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
            processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

            # TODO move (during the refactor)
            if (d := args.dump_details) is not None:
                tokenizer.save_pretrained(Path(d) / "tokenizer")
                if processor:
                    processor.save_pretrained(Path(d) / "processor")

            self.dataset = Dataset(
                args.prompt_data,
                tokenizer=tokenizer,
                processor=processor,
                max_length=args.rollout_max_prompt_len,
                prompt_key=args.input_key,
                multimodal_keys=args.multimodal_keys,
                label_key=args.label_key,
                metadata_key=args.metadata_key,
                tool_key=args.tool_key,
                apply_chat_template=args.apply_chat_template,
                apply_chat_template_kwargs=args.apply_chat_template_kwargs,
                seed=args.rollout_seed,
            )
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)

            # Dynamic dataset support: immutable full dataset for adjust_dataset to read,
            # and for select_dataset to re-filter from.
            self.full_samples: list[Sample] = list(self.dataset.origin_samples)
            self._last_difficulty_mapping: dict[int, int] | None = None
        else:
            self.dataset = None
            self.full_samples = []
            self._last_difficulty_mapping = None

    def get_samples(self, num_samples):
        # TODO further improve code
        if self.dataset is not None:
            if self.sample_offset + num_samples <= len(self.dataset):
                prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
                self.sample_offset += num_samples
            else:
                prompt_samples = self.dataset.samples[self.sample_offset :]
                num_samples -= len(prompt_samples)
                self.epoch_id += 1
                if self.args.rollout_shuffle:
                    self.dataset.shuffle(self.epoch_id)
                prompt_samples += self.dataset.samples[:num_samples]
                self.sample_offset = num_samples
        else:
            prompt_samples = [Sample() for _ in range(num_samples)]

        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
        return samples

    def select_dataset(self, difficulty_mapping: dict[int, int],
                       keep_min: int = 1, keep_max: int = 6) -> dict:
        """
        Filter the training dataset based on difficulty scores.

        Retains only samples whose difficulty falls within [keep_min, keep_max].
        Samples not present in difficulty_mapping are kept by default (unevaluated).

        Args:
            difficulty_mapping: Maps Sample.global_index to difficulty value
                (integer 0-8, number of successful rollouts out of 8).
            keep_min: Minimum difficulty to retain (inclusive). Default 1.
            keep_max: Maximum difficulty to retain (inclusive). Default 6.

        Returns:
            A report dict with statistics about the adjustment.
        """
        if self.dataset is None:
            logger.warning("select_dataset called but dataset is None. Skipping.")
            return {"skipped": True}

        self._last_difficulty_mapping = difficulty_mapping

        total_full = len(self.full_samples)
        total_before = len(self.dataset.origin_samples)
        prev_ids = set(id(s) for s in self.dataset.origin_samples)

        removed_easy = 0
        removed_hard = 0
        retained = 0
        unevaluated = 0
        restored = 0

        new_origin = []
        for sample in self.full_samples:
            gidx = sample.global_index
            if gidx not in difficulty_mapping:
                new_origin.append(sample)
                unevaluated += 1
                continue

            difficulty = difficulty_mapping[gidx]
            if difficulty > keep_max:
                removed_easy += 1
            elif difficulty < keep_min:
                removed_hard += 1
            else:
                new_origin.append(sample)
                if id(sample) not in prev_ids:
                    restored += 1
                else:
                    retained += 1

        # Safety: if ALL samples would be removed, keep dataset unchanged
        if len(new_origin) == 0:
            logger.warning(
                "select_dataset: filtering would remove ALL samples. "
                "Keeping dataset unchanged."
            )
            return {
                "total_full": total_full,
                "total_before": total_before,
                "total_after": total_before,
                "removed_easy": 0,
                "removed_hard": 0,
                "retained": total_before,
                "restored": 0,
                "unevaluated": 0,
                "warning": "all_removed_fallback",
            }

        # Update BOTH origin_samples and samples to avoid the shuffle trap:
        # Dataset.shuffle() uses len(self.samples) for permutation but indexes
        # into origin_samples, so both must have the same filtered content.
        self.dataset.origin_samples = new_origin
        self.dataset.samples = list(new_origin)
        self.dataset.epoch_id = -1  # force next shuffle to execute

        self.sample_offset = 0

        if self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)

        report = {
            "total_full": total_full,
            "total_before": total_before,
            "total_after": len(self.dataset.origin_samples),
            "removed_easy": removed_easy,
            "removed_hard": removed_hard,
            "retained": retained,
            "restored": restored,
            "unevaluated": unevaluated,
        }
        logger.info(f"select_dataset: {report}")
        return report

    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return

        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
            "last_difficulty_mapping": self._last_difficulty_mapping,
        }
        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset:
            return

        if self.args.load is None:
            return

        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            logger.info(f"Checkpoint {path} does not exist.")
            return

        logger.info(f"load metadata from {path}")
        logger.info(f"load metadata: {self.metadata}")
        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})

        # Restore dynamic dataset state by replaying the last filtering operation.
        self._last_difficulty_mapping = state_dict.get("last_difficulty_mapping", None)
        if self._last_difficulty_mapping is not None:
            self.select_dataset(self._last_difficulty_mapping)
            # select_dataset resets sample_offset to 0; restore the exact checkpoint offset.
            self.sample_offset = state_dict.get("sample_offset", 0)
        elif self.args.rollout_global_dataset and self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)


class RolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = []
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        samples += super().get_samples(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if len(self.buffer) == 0 or num_samples == 0:
            return []

        samples = self.buffer_filter(self.args, None, self.buffer, num_samples)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        """
        Add a sample group to buffer.
        """
        if not samples:
            return
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"
        for i in range(0, len(samples)):
            assert (
                len(samples[i]) == self.args.n_samples_per_prompt
            ), f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.args.n_samples_per_prompt}"
            group = samples[i]  # type: ignore
            self.buffer.append(group)

    # TODO remove
    def update_metadata(self, metadata: dict):
        self.metadata.update(metadata)

    # TODO remove
    def get_metadata(self):
        return self.metadata

    def get_buffer_length(self):
        return len(self.buffer)


def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
