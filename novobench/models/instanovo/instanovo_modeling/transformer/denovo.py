import logging
import torch
from tqdm import tqdm
import os
import pandas as pd
from novobench.models.instanovo.instanovo_dataloader import InstanovoDataModule
from novobench.models.instanovo.instanovo_modeling.constants import MASS_SCALE
from novobench.models.instanovo.instanovo_modeling.inference.knapsack import Knapsack
from novobench.models.instanovo.instanovo_modeling.inference.knapsack_beam_search import KnapsackBeamSearchDecoder
from novobench.models.instanovo.instanovo_modeling.transformer.model import InstaNovo

logger = logging.getLogger()
logger.setLevel(logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _setup_knapsack(model) -> Knapsack:
    residue_masses = model.peptide_mass_calculator.masses
    residue_masses["$"] = 0
    residue_indices = model.decoder._aa2idx
    return Knapsack.construct_knapsack(
        residue_masses=residue_masses,
        residue_indices=residue_indices,
        max_mass=4000.00,
        mass_scale=MASS_SCALE,
    )

def denovo_instanovo(test_df, config, model_filename, output_path):
    # dataloader
    if config['instanovo']["dec_type"] != "depthcharge":
        vocab = ["PAD", "<s>", "</s>"] + list(config['instanovo']["residues"].keys())
    else:
        vocab = list(config['instanovo']["residues"].keys())
    config["vocab"] = vocab
    s2i = {v: i for i, v in enumerate(vocab)}
    i2s = {i: v for i, v in enumerate(vocab)}
    logging.info(f"Vocab: {i2s}")

    dl = InstanovoDataModule(
        df = test_df,
        s2i = s2i,
        return_str = True,
        batch_size = config["predict_batch_size"],
        n_workers = config["n_workers"]
    ).get_dataloader()

    # model
    model, _ = InstaNovo.load(model_filename)
    model = model.to(device).eval()

    knapsack_path = config['instanovo']["knapsack_path"]
    if not os.path.exists(knapsack_path):
        print("Knapsack path missing or not specified, generating...")
        knapsack = _setup_knapsack(model)
        decoder = KnapsackBeamSearchDecoder(model, knapsack)
        print(f"Saving knapsack to {knapsack_path}")
        knapsack.save(knapsack_path)
    else:
        print("Knapsack path found. Loading...")
        decoder = KnapsackBeamSearchDecoder.from_file(model=model, path=knapsack_path)

    preds = []
    targs = []
    probs = []

    for _, batch in tqdm(enumerate(dl), total=len(dl)):
        spectra, precursors, _, peptides, _ = batch
        spectra = spectra.to(device)
        precursors = precursors.to(device)

        with torch.no_grad():
            p = decoder.decode(
                spectra=spectra,
                precursors=precursors,
                beam_size=config['instanovo']["n_beams"],
                max_length=config['instanovo']["max_length"],
            )

        preds += ["".join(x.sequence) if not isinstance(x, list) else "" for x in p]
        probs += [x.log_probability if not isinstance(x, list) else -1 for x in p]
        targs += list(peptides)
    
    # save results

    results = pd.DataFrame({
        "peptides_true": targs,
        "peptides_pred": preds,
        "peptides_score": probs
    })

    results.to_csv(output_path, index=False)
    logging.info(f"Results saved to {output_path}")





