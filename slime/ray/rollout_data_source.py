import copy
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from slime.utils.data import Dataset
from slime.utils.misc import load_function
from slime.utils.types import Sample


# TODO may further refactor data-loading part later
class RolloutDataSource:
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_index = 0
        self.sample_offset = 0
        # TODO remove this
        self.metadata = {}

        if args.rollout_global_dataset:
            tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

            # TODO move (during the refactor)
            if (d := args.dump_details) is not None:
                tokenizer.save_pretrained(Path(d) / "tokenizer")

            self.dataset = Dataset(
                args.prompt_data,
                tokenizer=tokenizer,
                max_length=args.rollout_max_prompt_len,
                prompt_key=args.input_key,
                label_key=args.label_key,
                metadata_key=args.metadata_key,
                tool_key=args.tool_key,
                apply_chat_template=args.apply_chat_template,
                seed=args.rollout_seed,
            )
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
        else:
            self.dataset = None

    def get_samples(self, num_samples):
        samples = []

        # TODO unify the two branches
        if self.dataset is not None:
<<<<<<< HEAD
            # self.sample_offset tracks the number of samples we have read from the dataset
            # we need to check if we have enough samples before reading
            if self.sample_offset + num_samples <= len(self.dataset):
                # if the remaining samples are enough, we can just read them
                prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
                # update the offset
                self.sample_offset += num_samples
            else:
                # we don't have enough samples, first read all remaining samples
                prompt_samples = self.dataset.samples[self.sample_offset :]
                # count how many samples left to reach num_samples
                num_samples -= len(prompt_samples)
                # we have used all samples in this epoch, update epoch_id
                self.epoch_id += 1
                # enter next epoch, first shuffle the dataset if args.rollout_shuffle is True
                if self.args.rollout_shuffle:
                    self.dataset.shuffle(self.epoch_id)
                
                # now get samples from the (newly shuffled) dataset
                prompt_samples += self.dataset.samples[:num_samples]
                # set the offset to the number of samples we have used
                self.sample_offset = num_samples
                
            # next we repeat samples according to args.n_samples_per_prompt
            for prompt_sample in prompt_samples:
                group = []
                for _ in range(self.args.n_samples_per_prompt):
                    # deepcopy sample and track the sample index
=======
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
            for prompt_sample in prompt_samples:
                group = []
                for _ in range(self.args.n_samples_per_prompt):
>>>>>>> origin/main
                    sample = copy.deepcopy(prompt_sample)
                    sample.index = self.sample_index
                    self.sample_index += 1
                    group.append(sample)
                samples.append(group)
        else:
<<<<<<< HEAD
            # cases where we don't have a dataset, use placeholders.
=======
>>>>>>> origin/main
            for _ in range(num_samples):
                group = []
                for _ in range(self.args.n_samples_per_prompt):
                    sample = Sample(
                        index=self.sample_index,
                    )
                    self.sample_index += 1
                    group.append(sample)
                samples.append(group)

        return samples

    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return

        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
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
            print(f"Checkpoint {path} does not exist.")
            return

        print(f"load metadata from {path}")
        print(f"load metadata: {self.metadata}")
        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})

        if self.args.rollout_global_dataset and self.args.rollout_shuffle:
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
<<<<<<< HEAD
        Return num_samples samples，partially rollout
        """
        
        # first try to get samples from the buffer
        samples = self._get_samples_from_buffer(num_samples)
        # print(num_samples, len(samples))
        # exit(0)
        num_samples -= len(samples)
        
        # if target reached, return
        if num_samples == 0:
            return samples
        
        # else draw samples from the data source.
=======
        Return num_samples samples
        """

        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

>>>>>>> origin/main
        samples += super().get_samples(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
<<<<<<< HEAD
        """Get samples from buffer, the behavior is controlled by buffer_filter,
        returns empty list if buffer if empty or num_samples == 0.

        Args:
            num_samples (int): _description_

        Returns:
            list[list[Sample]]: _description_
        """
=======
>>>>>>> origin/main
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
<<<<<<< HEAD
            assert (  # check whether each group has the same number of samples
=======
            assert (
>>>>>>> origin/main
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
<<<<<<< HEAD
    """Get first num_samples samples and remove them from buffer.
    if exceed buffer length, return all samples.

    Args:
        args (Namespace): arguments
        rollout_id (int): rollout id
        buffer (list[list[Sample]]): the buffer
        num_samples (int): number of samples

    Returns:
        list[list[Sample]]: first num_samples samples
    """
=======
>>>>>>> origin/main
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
