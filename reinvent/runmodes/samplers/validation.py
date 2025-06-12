"""Config Validation"""

from typing import Optional

from reinvent.validation import GlobalConfig, Field


class SectionParameters(GlobalConfig):
    model_file: str
    num_smiles: int
    smiles_file: Optional[str] = None
    target_smiles_path: str = ""
    sample_strategy: Optional[str] = "multinomial"  # Mol2Nol
    output_file: str = "samples.csv"
    target_nll_file: str = "target_nll.csv"
    unique_molecules: bool = True
    randomize_smiles: bool = True
    temperature: float = 1.0
    # UQ edit:
    dropout_prob: float = 0.0
    mode: str = 'training'
    sampling_mode: str = 'random'

class SectionResponder(GlobalConfig):
    endpoint: str
    frequency: Optional[int] = Field(1, ge=1)


class SamplingConfig(GlobalConfig):
    parameters: SectionParameters
    responder: Optional[SectionResponder] = None
