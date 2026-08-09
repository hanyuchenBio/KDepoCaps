# KDepoCaps

**KDepoCaps** is a machine learning tool for predicting the capsular types targeted by *Klebsiella pneumoniae* phage depolymerases from protein sequences.

## Installation

Clone the repository:

```bash
git clone https://github.com/hanyuchenBio/KDepoCaps.git
cd KDepoCaps
```

Create and activate the Conda environment:

```bash
conda env create -f KDepoCaps_environment.yml
conda activate KDepoCaps
```

## Download ESM-C

KDepoCaps uses **ESM-C 300M** to generate protein embeddings.

Download the model weights:

[Download ESM-C 300M](https://huggingface.co/biohub/esmc-300m-2024-12/resolve/main/data/weights/esmc_300m_2024_12_v0.pth?download=true)

or:

```bash
wget -O esmc_300m_2024_12_v0.pth \
"https://huggingface.co/biohub/esmc-300m-2024-12/resolve/main/data/weights/esmc_300m_2024_12_v0.pth?download=true"
```

The trained KDepoCaps model (`protein_model.joblib`) is already included in this repository.

## Input

Protein sequences in FASTA format:

```text
>Depolymerase_1
MNNNKDLIELSKKLE...
>Depolymerase_2
MSTNKIAVIGGGDS...
```

A single FASTA file or a directory containing FASTA files can be used as input.

## Usage

```bash
python KDepoCaps.py \
    -i proteins.fasta \
    -esm esmc_300m_2024_12_v0.pth \
    -pm protein_model.joblib \
    -o prediction.csv
```

### Parameters

| Option | Description                           |
| ------ | ------------------------------------- |
| `-i`   | Input protein FASTA file or directory |
| `-esm` | ESM-C 300M model path                 |
| `-pm`  | Trained KDepoCaps model               |
| `-o`   | Output prediction table               |

## Output

The output contains the prediction score of each protein for each KL type:

```text
ID,KL1,KL2,KL10,KL24,...
Depolymerase_1,0.03,0.87,0.11,0.02,...
Depolymerase_2,0.82,0.07,0.05,0.11,...
```

Each KL type is predicted independently by a binary Random Forest model.

Higher scores indicate a stronger predicted association between the depolymerase and the corresponding capsular type.

## Workflow

```text
Protein sequence
      ↓
   ESM-C 300M
      ↓
Protein embedding
      ↓
KL-specific Random Forest models
      ↓
Capsular-type prediction scores
```

## Help

```bash
python KDepoCaps.py -h
```

## Citation

If you use KDepoCaps in your research, please cite the corresponding publication.

Citation information will be added after publication.
