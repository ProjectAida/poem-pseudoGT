# Integrating Textual and Visual Features for Pseudo-Groundtruth Generation in Document Image Classification

This repository contains the code associated with the manuscript titled:  
**"Integrating Textual and Visual Features to Generate Pseudo-Groundtruth for Enhancing Training of Machine Learning Models in Document Image Classification."**

In this work, we propose a method that leverages both textual features—extracted from noisy OCR outputs—and visual features—derived from document image layouts—to generate pseudo-groundtruth labels for training deep learning models.

We construct three base classifiers:
- **Textual Base Classifier**
- **Visual Base Classifier**
- **Textual-Visual Base Classifier**

Each model predicts intermediate pseudo-labels for unlabeled samples. These labels are then integrated using data programming techniques to produce final pseudo-groundtruth labels. We evaluate three such techniques:
1. **Averaging**
2. **Merge-Layer**
3. **Snorkel**

This approach aims to improve training effectiveness when manually labeled data is limited or unavailable.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Citation] (#citation)
- [License](#license)

## Installation

Instructions on how to install and set up the project.

```bash
# Clone the repository
git clone https://github.com/ProjectAida/poem-pseudoGT.git
cd poem-pseudoGT

# (Optional) Create and activate a conda environment
conda env create --name poemPGT --file=environments.yml
conda activate poemPGT
```

## Usage
Each script trains a document image classification model using one of the three pseudo-groundtruth generation strategies: Averaging, Merge-Layer, or Snorkel. The general usage format is:

```bash
# python <python script> <experiment name> <running times>
# e.g.,

# Run Averaging 
python PGTavg/CNN_w_psudoGT.py "pgt_avg_3_0" 3
python PGTavg/CNN_w_psudoGT_rfinetune.py "pgt_avg_3_0" 3

# Run Merge-Layer
python PGTmrg/pgt_gen_run_merge.py "pgt_mrg_rft_3_0" 3
python PGTmrg/pgt_gen_run_merge_rfinetune.py "pgt_mrg_rft_3_0" 3

# Run Snorkel
python PGTsnk/pgt_gen_run_snorkel.py "pgt_snk_rft_3_0" 3
python PGTsnk/pgt_gen_run_snorkel_rfinetune.py "pgt_snk_rft_3_0" 3
```

## Citation

If you use this code or dataset in your work, please cite the associated paper:

```bibtex
@article{liu2025pseudoGT,
  title={Integrating Textual and Visual Features to Generate Pseudo-Groundtruth for Enhancing Training of Machine Learning Models in Document Image Classification},
  author={Liu, Yi and Soh, Leen-Kiat},
  journal={Journal of Electronic Imaging},
  year={2025},
  note={In press}
}
```

## License

This project is licensed under the [MIT License](LICENSE).