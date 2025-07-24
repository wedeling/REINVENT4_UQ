"""Classical Reinvent de novo model

See:
https://doi.org/10.1186/s13321-017-0235-x (original publication)
https://doi.org/10.1021/acs.jcim.0c00915 (REINVENT 2.0)
"""

from __future__ import annotations
from typing import Tuple, TypeVar, Iterator, TYPE_CHECKING

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as tnn

from reinvent.models.reinvent.models import rnn, vocabulary as mv
from reinvent.models.reinvent.utils import collate_fn
from reinvent.models.model_mode_enum import ModelModeEnum

if TYPE_CHECKING:
    from reinvent.models.meta_data import ModelMetaData

M = TypeVar("M", bound="Model")


class Model:
    """
    Implements an RNN model using SMILES.
    """

    _model_type = "Reinvent_UQ"
    _version = 1

    def __init__(
        self,
        vocabulary: mv.Vocabulary,
        tokenizer: mv.SMILESTokenizer,
        meta_data: ModelMetaData,
        network_params: dict = None,
        max_sequence_length: int = 256,
        mode: str = "training",
        device=torch.device("cpu"),
    ):
        """
        Implements an RNN using either GRU or LSTM.

        :param vocabulary: vocabulary to use
        :param tokenizer: tokenizer to use
        :param meta_data: model meta data
        :param network_params: parameters required to initialize the RNN
        :param max_sequence_length: maximum length of sequence that can be generated
        :param mode: either "training" or "inference"
        :param device: the PyTorch device
        """

        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.tokens = self.vocabulary.tokens()
        self.meta_data = meta_data
        self.max_sequence_length = max_sequence_length
        self.device = device

        if not isinstance(network_params, dict):
            network_params = {}

        self._model_modes = ModelModeEnum()

        self.network = rnn.RNN(len(self.vocabulary), **network_params, device=self.device)
        self.set_mode(mode)

        self._nll_loss = tnn.NLLLoss(reduction="none")

    def set_mode(self, mode: str) -> None:
        """
        Set training or inference mode of the network.

        :param mode: Mode to be set.
        :raises ValueError: raised when unknown mode
        """
        if mode == self._model_modes.TRAINING:
            self.network.train()
        elif mode == self._model_modes.INFERENCE:
            self.network.eval()
        else:
            raise ValueError(f"Invalid model mode '{mode}")

    def set_sampling_mode(self, sampling_mode):
        self.sampling_mode = sampling_mode

    @classmethod
    def create_from_dict(cls: type[M], save_dict: dict, mode: str, device: torch.device) -> M:
        model_type = save_dict.get("model_type")

        if model_type and model_type != cls._model_type:
            raise RuntimeError(f"Wrong type: {model_type} but expected {cls._model_type}")

        if isinstance(save_dict["vocabulary"], dict):
            vocabulary = mv.Vocabulary.load_from_dictionary(save_dict["vocabulary"])
        else:
            vocabulary = save_dict["vocabulary"]

        model = cls(
            vocabulary=vocabulary,
            tokenizer=save_dict.get("tokenizer", mv.SMILESTokenizer()),
            meta_data=save_dict.get("metadata"),
            network_params=save_dict.get("network_params"),
            max_sequence_length=save_dict["max_sequence_length"],
            mode=mode,
            device=device,
        )

        model.network.load_state_dict(save_dict["network"])

        return model

    def get_save_dict(self):
        """Return the layout of the save dictionary"""

        save_dict = dict(
            model_type=self._model_type,
            version=self._version,
            metadata=self.meta_data,
            vocabulary=self.vocabulary.get_dictionary(),
            tokenizer=self.tokenizer,
            max_sequence_length=self.max_sequence_length,
            network=self.network.state_dict(),
            network_params=self.network.get_params(),
        )

        return save_dict

    def save(self, file_path: str) -> None:
        """Saves the model into a file

        :param file_path: Path to the model file.
        """

        save_dict = self.get_save_dict()
        torch.save(save_dict, file_path)

    save_to_file = save  # alias for backwards compatibility

    def likelihood_smiles(self, smiles: str) -> torch.Tensor:
        tokens = [self.tokenizer.tokenize(smile) for smile in smiles]
        encoded = [self.vocabulary.encode(token) for token in tokens]

        sequences = [
            torch.tensor(encode, dtype=torch.long, device=self.device) for encode in encoded
        ]
        padded_sequences = collate_fn(sequences)

        return self.likelihood(padded_sequences)

    def likelihood(self, sequences: torch.Tensor) -> torch.Tensor:
        """Retrieves the likelihood of a given sequence

        Used in training.

        :param sequences: a batch of sequences (batch_size, sequence_length)
        :returns: log likelihood for each example (batch_size)
        """

        logits, _ = self.network(sequences[:, :-1])  # all steps done at once
        log_probs = logits.log_softmax(dim=2)

        return self._nll_loss(log_probs.transpose(1, 2), sequences[:, 1:]).sum(dim=1)

    @torch.no_grad()
    def sample(self, batch_size: int = 1, n_samples: int = 1, 
               input_length: int = 1, input_smilies = None):
        """
        Sample REINVENT given some partial input SMILES string.

        Parameters
        ----------
        batch_size : int, optional
            The batch size, here equal to the number of input SMILES.
            The default is 1.
        n_samples : int, optional
            How many samples to draw per input SMILES. The default is 1.
        input_length : int, optional
            The number of tokens in the input SMILES. The default is 1.
        input_smilies : TYPE, optional
            If None, the input SMILES are generated randomly. Otherwise
            they are read from a user-specified file, given in the config
            toml (smiles_file). The default is None.

        Returns
        -------
        seqs : list
            List containing the integer input sequences. Each entry is
            a tensor of size (batch_size, ).
        smiles : list
            A list of the sampled SMILES strings
        nll : tensor
            Tensor containing the NLL values

        """

        # no user-specified input SMILES, generate these randomly
        if input_smilies is None:
            input_sequence = None
        # user-specified input SMILES
        else:
            # tokenize all input smiles            
            tokens = [self.tokenizer.tokenize(data=token) for token in input_smilies]
            # remove end token '$' (these are partial, unfinished smiles)
            [token.pop(-1) for token in tokens]

            # all user-specified input SMILES should have the same number of tokens
            token_lengths = np.array([len(t) for t in tokens])
            assert (token_lengths == token_lengths[0]).all(), \
                "All input SMILES in smiles_file must be of the same size"

            # the length of the user_specified input SMILES must = input_length
            # -1 because we do not count start token
            assert input_length == token_lengths[0] - 1, \
                "The length of the input SMILES must equal the input_length configuration parameter"

            # convert tokens to integer sequence
            input_sequence = []
            for i in range(input_length + 1):
                tmp = []
                for j in range(batch_size):
                    tmp.append(self.tokens.index(tokens[j][i]))
                input_sequence.append(torch.Tensor(tmp).long())

        # get the state of the network given the (random) input sequence
        sequences, input_vector, hidden_state, nlls = self._get_input(input_sequence, 
                                                                       batch_size,
                                                                       input_length)

        # sample the network multiple times conditional on the (random) input
        # sequence
        Smiles = np.empty((n_samples, batch_size), dtype=object)
        Likelihoods = torch.empty((n_samples, batch_size))
        for i in tqdm(range(n_samples)):

            seqs, likelihoods = self._sample(sequences, input_vector, hidden_state, nlls)

            smiles = [
                self.tokenizer.untokenize(self.vocabulary.decode(seq)) for seq in seqs.cpu().numpy()
            ]

            Smiles[i] = smiles
            Likelihoods[i] = likelihoods

        return seqs, Smiles.T.flatten(), Likelihoods.T.flatten()

    @torch.no_grad()
    def _sample(self, init_seq, init_input_vector, hidden_state, init_nlls):
        """
        Run the network forward given the state consistent with
        the input smiles, given by _get_input(...).

        Parameters
        ----------
        init_seq : list
            The integer sequences from the start token to the 
            second-to-last input token.
        init_input_vector : tensor
            The input vector of the last token of the input SMILES.
        hidden_state : tuple
            The hidden state of the network consistent with the
            input state.
        init_nlls : tensor
            The NLL values consistent with the inout state.


        Returns
        -------
        concat_sequences : tensor
            The integer sequences of the full SMILES strings. Shape
            is (smiles_length, batch_size).
        nlls : tensor
            The NLL values of the full SMILES strings. Shape is
            (batch_size, )

        """

        # make indepdent copies of the initial sequence, nlls and input vector
        # the hidden state does not get overwritten by calling this
        # subroutine multiple times
        sequences = [idx.detach().clone() for idx in init_seq]
        input_vector = init_input_vector.detach().clone()
        nlls = init_nlls.detach().clone()

        for _ in range(self.max_sequence_length - 1):
            logits, hidden_state = self.network(input_vector.unsqueeze(1), hidden_state)
            logits = logits.squeeze(1)  # 2D
            log_probs = logits.log_softmax(dim=1)  # 2D
            probabilities = logits.softmax(dim=1)  # 2D

            if self.sampling_mode == 'random':
                input_vector = torch.multinomial(probabilities, num_samples=1).view(-1)  # 1D
            elif self.sampling_mode == 'max': 
                input_vector = probabilities.argmax(dim=1).view(-1)
            else:
                raise ValueError(f"Invalid sampling mode: {self.sampling_mode}")
 
            sequences.append(input_vector.view(-1, 1))
            nlls += self._nll_loss(log_probs, input_vector)

            if input_vector.sum() == 0:
                break

        concat_sequences = torch.cat(sequences, dim=1)

        return concat_sequences.data, nlls

    @torch.no_grad()
    def _get_input(self, input_sequence, batch_size: int = 1, input_length: int = 1):
        """
        Run the network forward up to the partial input sequence.

        Parameters
        ----------
        input_sequence : list
            Either None or a list of user-specified input tensors of size 
            (bach_size, ). If none the inout sequence will be generated
            at random by simply sampling the softmax distribution.
        batch_size : int, optional
            The batch size, here equal to the number of input SMILES.
            The default is 1.
        input_length : int, optional
            The number of tokens in the input SMILES. The default is 1.

        Returns
        -------
        sequences : list
            A list of input tensors.
        input_vector : tensor
            The next input tensor, the last token of the input tensor.
        hidden_state : tuple
            The hidden state of the RNN.
        nlls : tensor
            Tensor of nll values.

        """

        # the first sequences are just the start token repeated batch_size times
        sequences = [
            torch.full(
                (batch_size, 1),
                self.vocabulary[mv.START_TOKEN],
                dtype=torch.long,
                device=self.device,
            )
        ]       

        # same for the input vector
        input_vector = torch.full(
            (batch_size,), self.vocabulary[mv.START_TOKEN], dtype=torch.long, device=self.device
        )
        
        # initial hidden state
        hidden_state = None
        # store nll values
        nlls = torch.zeros(batch_size, device=self.device)

        # loop from start token to second to last token
        for i in range(input_length):
            logits, hidden_state = self.network(input_vector.unsqueeze(1), hidden_state)
            logits = logits.squeeze(1)  # 2D
            probabilities = logits.softmax(dim=1)  # 2D
            log_probs = logits.log_softmax(dim=1)  # 2D

            # if no user-specified input SMILES, generate randomly
            if input_sequence is None:
                input_vector = torch.multinomial(probabilities, num_samples=1).view(-1)  # 1D
            else:
                input_vector = input_sequence[i+1]

            sequences.append(input_vector.view(-1, 1))
            nlls += self._nll_loss(log_probs, input_vector)

            if input_vector.sum() == 0:
                break

        return sequences, input_vector, hidden_state, nlls

 
    def get_network_parameters(self) -> Iterator[tnn.Parameter]:
        """
        Returns the configuration parameters of the network.

        :returns: network parameters of the RNN
        """

        return self.network.parameters()
                   
