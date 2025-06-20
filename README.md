# MPrediction

## Python Environment Install

conda create -n myenv python=3.9

conda activate myenv

conda install scikit-learn

conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install dgl==2.0.0+cu118

conda install rdkit

## Usage

### Data Preprocess

Pre-trained data: we download the orthopoxvirus-related bioassay data from PubChem database, remove the redundant bioassay data and obtain the final 99621 compounds (79581, 17548 and 492 respectively)

Fine-tuned data: we download and filter the compounds with anti-orthopoxvirus bioactivity (such as vaccinia virus, monkeypox virus, cowpox virus) from ChEMBL database. At the same time, we combine 31 self-established BBR derivatives from the web experiments and finally obtain 1803 compounds for fine-tuning.

### Pre-training

```bash
python BERT_training.py
```

### Fine-tune and test

```bash
python fine-tune.py
```
## File descriptions

Pretrained Weight file: pretain2.pt

Dictionary file: ident_base_r2m.pickle
