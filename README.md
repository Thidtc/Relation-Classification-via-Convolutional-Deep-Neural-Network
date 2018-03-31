# Relation Classification via Convolutional Deep Neural Network

This is an implementation of paper [Relation Classification via Convolutional Deep Neural Network](http://www.aclweb.org/anthology/C14-1220) in PyTorch

## Model
![](./framework.png)

## Environment
* Python 3.*
* PyTorch 0.3.*
* fire 0.1.2

## Data
The experiment data is SemEval-2010 task8 data, and is available at [link](https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk/view?layout=list&ddrp=1&sort=name&num=50#)

## Usage
Data preprocess

```bash
python process.py --in_filename=ARGS --out_filename=ARGS
```

Training

```bash
python cnn_train.py --KEY=VALUE
```

## Result
Accuracy
67.84%

## TODO
parameters fine-tune, so as to improve the accuracy