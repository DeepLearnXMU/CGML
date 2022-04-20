# ML-TranX and ML-TranX'
These two models are based on a general-purpose **Tran**sition-based abstract synta**X** parser [ACL '18 paper](https://arxiv.org/abs/1806.07832) and  
**TranX'** is a variant of **TranX** which decodes in a breadth-first manner. By adopting a mutual learning based model training framework, both models can fully absorb the knowledge from each other and thus could be improved simultaneously.


## System Architecture
The System Architecture is the same as **TranX** [ACL '18 paper](https://arxiv.org/abs/1806.07832). Please refer to https://github.com/pcyin/tranX .


## Usage

```bash
cd tranX
conda env create -f config/env/tranx-py2.yml
./scripts/django/train-mutual-share-embedding.sh 0  # train on django code generation dataset  with random seed 0

conda env create -f config/env/tranx.yml  # create conda Python environment. 
./scripts/atis/train-mutual-share-embedding.sh 0  # train on ATIS semantic parsing dataset
./scripts/geo/train-mutual-share-embedding.sh 0  # train on GEO dataset
./scripts/ifttt/train-mutual-share-embedding.sh 0  # train on IFTTT dataset

./scripts/django/test_used.sh    # modify the configuration to test the model on django code generation dataset
./scripts/atis/test_used.sh     # modify the configuration to test the model on atis code generation dataset
./scripts/geo/test_used.sh      # modify the configuration to test the model on geo code generation dataset
./scripts/ifttt/test_used.sh     # modify the configuration to test the model on ifttt code generation dataset
```
