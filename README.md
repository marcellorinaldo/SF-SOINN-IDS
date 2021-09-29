# SF-SOINN-IDS
An Intrusion Detection System (IDS) based on a modified version of the Self-Organizing Incremental Neural Networks, called Soft-Forgetting SOINN (SF-SOINN).

Contents:
* `SOINN` folder contains the SF-SOINN Python class
* `nsl-kdd.ipynb` is the Python notebook with the experiments on the NSL-KDD intrusion dataset
* `cic-ids-2017.ipynb` is the Python notebook with the experiments on the CIC-IDS-2017 intrusion dataset
* `test-data` folder contains a Python notebook, data_generator.ipynb, that creates the pickles of artificial 2D datasets
* `artificial_tests.ipynb` is a Python notebook with clustering experiments on the 2D artificial datasets

## Datasets

The datasets needed to run the notebooks are available at the following links:
* [Canadian Institute for Cybersecurity - NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)
* [Canadian Institute for Cybersecurity - CIC-IDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html)

## Note about the implementation
Implementation of the SF-SOINN network is done using [igraph](https://igraph.org/python/) due to its simplicity. It is not the most performing implementation, some little improvements could make the network's operations faster, and maybe other packages like [NetworkX](https://networkx.org/) perform better. If performances are more important than the simplicity of the code, then the network could be implemented using [Tensorflow](https://www.tensorflow.org/) to take advantage of the processing power of GPUs.
