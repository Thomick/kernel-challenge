# Kernel Challenge

## Team

- Polina Barabanshchikova
- Thomas MICHEL

## Challenge

This repository contains the code for the Kernel Challenge realized in the context of the course Kernel Methods for Machine Learning of the MVA master at ENS Paris-Saclay.

The goal of the challenge is to predict the labels of a test set of images given a training set of images. The images are 32x32 pixels and the labels are between 0 and 9. The training set contains 5000 images and the test set contains 2000 images.

Please refer to [our report](report.pdf) for more details on the methods used and the results obtained.

## Usage

First install the required python libraries:
```bash
pip install -r requirements.txt
```

Then you can run the code by executing the following command:
```bash
cd src
python start.py
```
This code will produce a file `Yte.csv` in the same directory as the `start.py` script directory containing the predicted labels for the test set. You can specify whether you want to retrain the models by setting the corresponding flags directly in the code.

Note that the scripts ending with `_submission.py` were also used to generate the final submission files using the other models we tried.