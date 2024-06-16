import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger("adanovo")

class AdanovoConfig:
    """
    configuration class for Adanovo.

    """

    def __init__(
        self,
        random_seed: int = 454,
        n_peaks: int = 150,
        min_mz: float = 50.0,
        max_mz: float = 2500.0,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 2.0,
        max_charge: int = 10,
        precursor_mass_tol: float = 50,
        isotope_error_range: Tuple[int, int] = (0, 1),
        min_peptide_len: int = 6,
        predict_batch_size: int = 1024,
        n_beams: int = 5,
        top_match: int = 1,
        accelerator: str = "auto",
        devices: Optional[int] = None,
        # Additional parameters for training new Adanovo models
        dim_model: int = 512,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 9,
        dropout: float = 0.0,
        dim_intensity: Optional[int] = None,
        max_length: int = 100,
        n_log: int = 1,
        tb_summarywriter: Optional[str] = None,
        train_label_smoothing: float = 0.01,
        warmup_iters: int = 100_000,
        max_iters: int = 600_000,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-5,
        train_batch_size: int = 32,
        max_epochs: int = 30,
        # max_epochs: int = 8,
        num_sanity_val_steps: int = 0,
        save_top_k: int = 5,
        model_save_folder_path: str = "/usr/commondata/local_public/jingbo/adanovo/nine_species_2/",
        val_check_interval: int = 50_000,
        calculate_precision: bool = False,
        n_workers: int = 64,
        s1: float = 0.1,
        s2: float = 0.1,
        **kwargs
    ):
        """Initializes AdanovoConfig with default or user-provided values."""
        self.random_seed = random_seed
        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.min_intensity = min_intensity
        self.remove_precursor_tol = remove_precursor_tol
        self.max_charge = max_charge
        self.precursor_mass_tol = precursor_mass_tol
        self.isotope_error_range = isotope_error_range
        self.min_peptide_len = min_peptide_len
        self.predict_batch_size = predict_batch_size
        self.n_beams = n_beams
        self.top_match = top_match
        self.accelerator = accelerator
        self.devices = devices
        self.dim_model = dim_model
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
        self.n_layers = n_layers
        self.dropout = dropout
        self.dim_intensity = dim_intensity
        self.max_length = max_length
        self.n_log = n_log
        self.tb_summarywriter = tb_summarywriter
        self.train_label_smoothing = train_label_smoothing
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_batch_size = train_batch_size
        self.max_epochs = max_epochs
        self.num_sanity_val_steps = num_sanity_val_steps
        self.save_top_k = save_top_k
        self.model_save_folder_path = model_save_folder_path
        self.val_check_interval = val_check_interval
        self.calculate_precision = calculate_precision
        self.residues = mass_dict
        self.n_workers = n_workers
        self.s1 = s1
        self.s2 = s2

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        attrs = {attr: getattr(self, attr) for attr in dir(self) if not attr.startswith("__") and not callable(getattr(self, attr))}
        return f"AdanovoConfig({attrs})"


mass_dict = {
  "G": 57.021464,
  "A": 71.037114,
  "S": 87.032028,
  "P": 97.052764,
  "V": 99.068414,
  "T": 101.047670,
  "C(+57.02)": 160.030649,
  "L": 113.084064,
  "I": 113.084064,
  "N": 114.042927,
  "D": 115.026943,
  "Q": 128.058578,
  "K": 128.094963,
  "E": 129.042593,
  "M": 131.040485,
  "H": 137.058912,
  "F": 147.068414,
  "R": 156.101111,
  "Y": 163.063329,
  "W": 186.079313,
  "M(+15.99)": 147.035400,
  "N(+.98)": 115.026943,
  "Q(+.98)": 129.042594,
}




