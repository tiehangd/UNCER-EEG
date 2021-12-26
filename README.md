
# UE-EEG

[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.4.0-%237732a8)](https://pytorch.org/get-started/previous-versions/)


Code for UE-EEG model introduced in ["Uncertainty Detection for EEG Neural Decoding Models"]()

## Setting up environment

The model is implemented with Pytorch, we recommend python 3.6 and PyTorch 1.4.0 with Anaconda.
    
Create a new environment and install python packages in it:

    conda create --name ue python=3.6
    conda activate ue
    conda install pytorch=1.4.0 -c pytorch
    conda install torchvision=0.5.0 -c pytorch
    conda install scipy scikit-learn

Clone the repository:
   
    git clone https://github.com/tiehangd/UE-EEG

## Dataset preparation

   Download BCI-IV 2a dataset from http://bnci-horizon-2020.eu/database/data-sets, Four class motor imagery (001-2014)
   
   Place the 18 files inside ./data folder
   
   Data preprocess, go into the ./dataloader folder, and run from command line
   
       python data_preprocessing_cross_subject.py
   
   for cross subject tasks
   
       python data_preprocessing_intra_subject.py
   
   for intra subject tasks
   
   The produced data files are stored in ./data/cross_sub and ./data/intra_sub

## Running the model

1) Model Training
    
        python train.py --model_name eegnet

2) Uncertainty Evaluation
    
        python eval.py -r -b -m \
            --load_model_name eegnet \
            --test_model_name eegnet_dropout \
            --p 0.02 \
            --min_variance 1e-3 \
            --noise_variance 1e-3 \
            --num_samples 100


## Citation

Please cite our paper if it is helpful to your work:
```
@article{Duan2021,
}
```


## Acknowledgements

  Implementation of UE-EEG model utilized code from the following repositories:
    
    1) https://github.com/mattiasegu/uncertainty_estimation_deep_learning
    2) https://github.com/aliasvishnu/EEGNet


