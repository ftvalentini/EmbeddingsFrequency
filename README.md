

Code to replicate [_Investigating the Frequency Distortion of Word Embeddings and Its Impact on Bias Metrics_](https://arxiv.org/abs/2211.08203) (Valentini et al., Findings EMNLP 2023).

Cite as:

```bibtex
@inproceedings{valentini-etal-2023-pmi,
    title = "{I}nvestigating the {F}requency {D}istortion of {W}ord {E}mbeddings and {I}ts {I}mpact on {B}ias {M}etrics",
    author = "Valentini, Francisco  and
      Sosa, Juan Cruz  and
      Fernandez Slezak, Diego  and
      Altszyler, Edgar",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
}
```

The following guide was run in Ubuntu 20.04.5 LTS with python=3.9.12 and R=4.2.3. You can set up a [conda environment](#conda-environment). 

## Requirements

Install **Python requirements**:

```
python -m pip install -r requirements.txt
```

Install **R requirements**:

```
Rscript install_packages.R
```

Clone [Stanford](https://nlp.stanford.edu/)'s GloVe repo into the repo:

```
git clone https://github.com/stanfordnlp/GloVe.git
```

or alternatively add it as submodule:

```
git submodule add https://github.com/stanfordnlp/GloVe
```

To build GloVe:

* In Linux: `cd GloVe && make`

* In Windows: `make -C "GloVe"`


## Guide

Follow steps in `full_pipeline.sh`. 


## conda environment

You can create a `we-frequency` conda environment to install requirements and dependencies. This is not compulsory. 

To install miniconda if needed, run:

```
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh 
sha256sum Miniconda3-py39_4.12.0-Linux-x86_64.sh 
bash Miniconda3-py39_4.12.0-Linux-x86_64.sh 
# and follow stdout instructions to run commands with `conda`
```

To create a conda env with Python:

```
conda config --add channels conda-forge
conda create -n "we-frequency" --channel=defaults python=3.9.12
```

Activate the environment with `conda activate we-frequency`. If `pip` is not installed, run `conda install pip`.
