A critical look at the identifiability of causal effect using deep latent variable models - Supplementary Material

The actual Supplementary Material PDF is the CriticalLookAtCausalIdentifiabilityDLVM_supplementary.pdf. 

The code:

The project is structured into notebooks that run the experiments and plot the results, and to Python files containing
most of the actual code. Most of the experiments use the definition of CEVAE in CEVAE.py, which is a very flexible
approach. In the MNIST experiment, we use a different Pytorch model defined in imageCEVAE.py. 

The seven experiments are in the following files (Corresponding to the order in the paper):
-running_lineargaussian_data.ipynb
-running_binary_data.ipynb
-running_irrelevantnoise_data.ipynb
-running_copyproxy_data.ipynb
-running_IHDP_data.ipynb
-running_MNIST_data.ipynb
-running_Twins_data.ipynb

The Python file cevaetools.py contains lots of the most relevant code for the experiments, and imagedatatools.py
as well for the MNIST data. The other Python files may be referenced in some specific parts of code. In particular,
the binarytoydata.py, lineartoydata.py, imagedata.py, contain code for generating different data sets. datagenVAE.py
and GANmodel.py contain the models for generating new data for the IHDP and MNIST experiments. 

I didn't include the actual data or most of the trained models, as those take up lots of space, but the folders
GANmodels and datageneratormodels contain pretrained models for generating mode MNIST and IHDP data, respectively.
The results of the experiments are saved in the data/ folder, which contains the data generating parameters and 
such used in the experiments, if they are not written out in the notebooks. 