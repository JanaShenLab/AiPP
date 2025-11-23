# AiPP: Artificial Intelligence Protein Profiling

Companion repository for:
-  Dayhoff II, Guy W., et al. "Illuminating the Druggable Human Proteome with an AI Protein Profiling Platform." bioRxiv (2025): 2025-09. (https://www.biorxiv.org/content/10.1101/2025.09.07.670677v2)

## How to cite AiPP

If you use the AiPP models, command-line interface, or web-based atlas
in your work, please cite:

  Dayhoff II, Guy W., Daniel Kortzak, Ruibin Liu, Mingzhe Shen,
  Zhong-Yin Zhang, and Jana Shen. "Illuminating the Druggable Human
  Proteome with an AI Protein Profiling Platform." bioRxiv (2025).

BibTeX:

  @article{dayhoff2025illuminating,
    title={Illuminating the Druggable Human Proteome with an AI Protein Profiling Platform},
    author={Dayhoff, Guy W and Kortzak, Daniel and Liu, Ruibin and Shen, Mingzhe and Zhang, Zhong-Yin and Shen, Jana},
    journal={bioRxiv},
    pages={2025--09},
    year={2025},
    publisher={Cold Spring Harbor Laboratory}
  }

## License ## 
**License** Content © 2025 Guy W. Dayhoff II and Jana Shen, licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

Unless otherwise noted, this repository’s content is © 2025 Guy W. Dayhoff II and Jana Shen (on behalf of all authors) and is licensed under Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0).

Plain English: You may copy, adapt, and share the repository’s content for non-commercial purposes as long as you provide proper attribution. For any commercial use, please contact the authors for permission.

# Web-based Inference and Atlas Exploration

We provide a browser-based AiPP interface for both on-demand
predictions and interactive exploration of a precomputed human
proteome atlas:

    Home:      https://aipp.computchem.org
    Inference: https://aipp.computchem.org/#predict
    Atlas:     https://aipp.computchem.org/#explore

The web tools run the same AiPP models described in the manuscript and
use ESM-C (`esmc-6b-2024-12`) via the EvolutionaryScale Forge API for
protein embeddings where on-demand inference is required.

------------------------------------------------------------
Web-based inference: /#predict
------------------------------------------------------------

The `/#predict` view provides a simple front end to the AiPP inference
pipeline for user-supplied sequences.

Requirements:

- A modern web browser.
- An active ESM Forge API token issued by EvolutionaryScale.

The AiPP web interface does not issue Forge tokens. Each user must
obtain and manage their own token directly from EvolutionaryScale.

Obtaining an ESM Forge API token:

1. Visit the EvolutionaryScale Forge site:

       https://forge.evolutionaryscale.ai

2. Sign up or sign in with your account.
3. In the leftmost menu, under the API header select 'API Keys'
4. In the textbox with the 'API Key Name' placehold text, name your key, e.g. aipp
5. Create a new Forge API token and copy the token string.

Treat this token as a secret (similar to a password); do not publish
or commit it.

Using the token in the AiPP `/#predict` interface:

1. Open:

       https://aipp.computchem.org/#predict

2. Enter your protein sequence, UniProtID, or PDBID-CHAINID into the input box.
3. Paste your ESM Forge API token into the token field.
4. Submit to run predictions.

The Forge token is used only to obtain ESM-C embeddings for your
sequences via the Forge API; AiPP then applies the pre-trained AiPP
heads to produce residue-level predictions.

Please follow EvolutionaryScale’s terms of use and your institution’s
policies when requesting, storing, and using Forge API tokens.

------------------------------------------------------------
Precomputed human AiPP atlas: /#explore
------------------------------------------------------------

The `/#explore` view provides interactive access to a precomputed
AiPP “human atlas” of predictions across the druggable human proteome.

Key points:

- Predictions in this atlas were precomputed using the AiPP models and
  ESM-C embeddings described in the manuscript.
- Exploration of the atlas (searching, browsing, viewing scores) does
  not require an ESM Forge token, because no new embeddings or model
  inference are run client-side.
- The atlas is intended as a convenient starting point for exploring
  residue-level predictions and associated experimental data without
  installing local software.

Typical usage:

1. Open:

       https://aipp.computchem.org/#explore

2. Search or browse for a protein of interest by UniProtKB accession
   or gene name.
3. Inspect the per-residue AiPP scores (LigCys, LigBind, SSBind,
   ZNBind, etc.) and any additional annotations provided.

The atlas integrates three key types of information, which correspond
to sections in the web interface:

1. Experimental evidence (LigABPP)
   - Manually curated cysteine-directed activity-based protein
     profiling (ABPP) measurements from the LigABPP database (as
     described in the manuscript).
   - Provides both site-level and cluster-level experimental evidence
     of cysteine ligandability across multiple studies.
   - These data can be viewed alongside AiPP model scores to assess
     concordance between experimental measurements and model
     predictions.

2. Homologous residue clusters
   - Cysteine sites are grouped using a composite similarity score in
     protein language model (PLM) embedding space.
   - Each panel groups residues that are homologous under this
     embedding; all residues in a cluster are treated as a single unit
     and are never split across train / validation / test partitions.

3. Homology-linked protein clusters
   - Proteins are grouped so that they remain together in any
     train / validation / test split.
   - Groups are formed by (i) never splitting an individual protein
     across partitions and (ii) linking proteins that share at least
     one cysteine in the same PLM-based residue cluster.
   - All proteins in a group are therefore assigned to the same data
     partition, ensuring that closely related proteins and residues
     do not leak across splits.

The home page at:

    https://aipp.computchem.org

provides a consolidated entry point to both the prediction interface
and the human atlas, along with brief explanatory text about the AiPP
platform.

# Command-line Inference Interface 
Command-line interface for running AiPP residue-level predictions
(SSBind, LigBind, ZNBind, LigCys) on protein sequences using ESM-C
embeddings and pre-trained weights.

---

## Repository layout

- aippCLI.py
  Main command-line interface.

- env/wizard.sh
  Simple installation wizard that creates a Python virtual environment,
  installs dependencies, and downloads pretrained weights from Zenodo.

- env/wts/
  Default location for model weight directories. The CLI expects
  task-specific subdirectories here (e.g. ssbind_v1, ligbind_v1, etc.).

---

## Requirements

- NVIDIA GPU with >= 24GB VRAM (e.g. RTX 4090)
- 128 GB system memory
- Python 3.10 or newer
- POSIX-like environment (Linux / macOS)
- Packages:
  - numpy
  - torch
  - esm
  - tqdm
  - httpx
  - colorama

You can install these manually:

    pip install numpy torch esm tqdm colorama httpx

or use the provided wizard.

---

## Installation

Clone the repository:

    git clone https://github.com/wayyne/aippCLI.git
    cd aippCLI

Run the installation wizard:

    bash env/wizard.sh

The wizard:

- creates a virtual environment named "AiPP"
- activates it
- upgrades pip
- installs core Python dependencies
- is the place to add commands to download weights from Zenodo into
  env/wts/

To re-activate the environment later:

    source AiPP/bin/activate

---

## Weights

By default, aippCLI.py looks for model weights under:

    env/wts/

with task-specific subdirectories (for example):

- env/wts/ssbind_v1
- env/wts/ligbind_v1
- env/wts/znbind_v1
- env/wts/ligcysA_v1
- env/wts/ligcysS_v1

You can override the root directory for weights with:

    export AIPP_WTS_DIR=/path/to/wts

and the CLI will use that directory instead of env/wts/.

---

## Forge token

The CLI requires an ESM Forge token to compute ESM-C embeddings.

- To obtain an ESM Forge token see Web-based inference.

Token handling:

1. First run:
   - Provide a token via --forge-token, either as the raw string or a
     path to a file that contains the token.
   - The token is cached to a user-level file (by default:
     ~/.aipp_forge_token).

2. Subsequent runs:
   - If --forge-token is omitted, the cached token is used.
   - If no cached token exists, the CLI will prompt for one
     interactively (input is hidden) and then cache it.

You can override the cache location with:

    export AIPP_FORGE_TOKEN_FILE=/path/to/token_file

---

## Basic usage

Activate the environment:

    source AiPP/bin/activate

Run a single sequence:

    python aippCLI.py \
      --sequence "ACDEFGHIKLMNPQRSTVWY" \
      --id example1 \
      --forge-token /path/to/forge_token.txt

Run multiple sequences from a FASTA file:

    python aippCLI.py \
      --fasta proteins.fasta \
      --forge-token /path/to/forge_token.txt

If a token has already been cached, you can omit --forge-token and the
CLI will reuse the saved token or prompt for one if needed.

---

## Output

By default, per-residue predictions are printed as a tab-separated
table to standard output.

To write the table to a file:

    python aippCLI.py \
      --sequence "ACDEFGHIKLMNPQRSTVWY" \
      --id example1 \
      --out results.tsv

At the end of the run you will see:

    output written to: results.tsv

The first line of the file is a header:

    pos    AA    SSBind    SSBind_topN    LigBind    LigBind_topN ...
    ...

Each subsequent line corresponds to a residue position.

---

## Reproducibility

The CLI echoes the full command line used to invoke it immediately
after the splash screen. This makes it straightforward to record and
reproduce runs from logs or publications.

------------------------------------------------------------
# Archived artifacts (immutable, citable)

- **LigCys ensemble weights v1.0.0 (Zenodo)** — DOI: `10.5281/zenodo.17210112` 
- **LigBind ensemble weights v1.0.0 (Zenodo)** — DOI: `10.5281/zenodo.17210749`
 
- **ESMC embeddings used by LigCys v1.0.0 (Zenodo)** — DOI: `10.5281/zenodo.17210943`
- **ESMC embeddings used by LigBind v1.0.0 (Zenodo)** — DOI: `10.5281/zenodo.17204577`
 
- **Packed datasets used to train LigCys v1.0.0 (Zenodo)**  — DOI: `10.5281/zenodo.17193767`
- **Packed datasets used to train LigBind v1.0.0 (Zenodo)** — DOI: `10.5281/zenodo.17204149`
  
- **Annotated/complete structures used to construct LigBind3D v1.0.0 (Zenodo)** — DOI: `10.5281/zenodo.17209968`

