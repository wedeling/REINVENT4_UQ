"""Reinvent adapter"""

__all__ = ["Reinvent_UQAdapter"]
from typing import List

import torch

from .sample_batch import SampleBatch
from reinvent.models.model_factory.model_adapter import ModelAdapter


class Reinvent_UQAdapter(ModelAdapter):
    def likelihood(self, sequences: torch.Tensor) -> torch.Tensor:
        return self.model.likelihood(sequences)

    def likelihood_smiles(self, smiles: List[str]) -> torch.Tensor:
        return self.model.likelihood_smiles(smiles)

    def sample(self, batch_size: int, n_samples: int, input_length: int, input_smilies = None) -> SampleBatch:
        # torch.Tensor, List[str], torch.Tensor
        sequences, smilies, nlls = self.model.sample(batch_size, n_samples, input_length, input_smilies)

        # NOTE: keep the sequences and nlls as Tensor as they are needed for
        #       later computations
        return SampleBatch(sequences, smilies, nlls)
