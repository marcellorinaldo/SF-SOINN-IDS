# SOINN-IDS
An Intrusion Detection System (IDS) based on a modified version of the Self-Organizing Incremental Neural Networks, called SF-SOINN.

Reference my Master's Degree thesis (`thesis.pdf`) for the theoretical details.

Contents:
* `SOINN` folder contains the SF-SOINN Python class.
* `NSL-KDD` folder contains the NSL-KDD intrusion dataset, also downloadable from [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/nsl.html).
* `nsl-kdd.ipynb` is the Python notebook with the experiments on the NSL-KDD intrusion dataset.
* `test-data` folder contains a Python notebook, data_generator.ipynb, that creates the pickles of artificial 2D datasets.
* `artificial_tests.ipynb` is a Python notebook with clustering experiments on the 2D artificial datasets.

## Note about the implementation
Implementation of the SF-SOINN network is done using [igraph](https://igraph.org/python/) due to its simplicity. It is not the most performing implementation, some little improvements could make the network's operations faster, and maybe other packages like [NetworkX](https://networkx.org/) perform better. If performances are more important than the simplicity of the code, then the network could be implemented using [Tensorflow](https://www.tensorflow.org/) to take advantage of the processing power of GPUs.
