# Mutate and Observe

This repository contains the code and some materials used in the experimental work presented in the following paper:

[1] [Mutate and Observe: Utilizing Deep Neural Networks to Investigate the Impact of Mutations on Translation Initiation](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btad338/7177993) <br> Bioinformatics, 2023.

## Data

All datasets used in the paper can be found in https://zenodo.org/record/7969148. In the /data folder, we provided a single sequence for the beta-globin gene as an example use-case.

## Model

The lightweight model (less than 1MB) which we called TISRover+ can be found under the model folder.

## Code

We share snippets of code to confirm the performance of the model used. The dataset and the model can be loaded with their respective file and be tested using main_inference.py. We are planning to make some of the code related to mutations available in the upcoming days (it takes a while to clean the code).

## Citation
If you find the code in this repository useful for your research, consider citing our paper.

    @article{10.1093/bioinformatics/btad338,
        author = {Ozbulak, Utku and Lee, Hyun Jung and Zuallaert, Jasper and De Neve, Wesley and Depuydt, Stephen and Vankerschaver, Joris},
        title = {Mutate and Observe: Utilizing Deep Neural Networks to Investigate the Impact of Mutations on Translation Initiation},
        journal = {Bioinformatics},
        year = {2023},
        month = {05},
        issn = {1367-4811},
        doi = {10.1093/bioinformatics/btad338},
        url = {https://doi.org/10.1093/bioinformatics/btad338}
    }

## Requirements
```
python > 3.5
torch >= 0.4.0
numpy >= 1.13.0
```
