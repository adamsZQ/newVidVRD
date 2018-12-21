## File description
  * `hyperparams.py` includes all hyper parameters that are needed.
  * `prepro.py` creates vocabulary files for the source and the target.
  * `data_load.py` contains functions regarding loading and batching data.
  * `modules.py` has all building blocks for encoder/decoder networks.
  * `train.py` has the model.
  * `eval.py` is for evaluation.

## Training
* STEP 1. Download [IWSLT 2016 German–English parallel corpus](https://wit3.fbk.eu/download.php?release=2016-01&type=texts&slang=de&tlang=en) and extract it to `corpora/` folder.
```sh
wget -qO- --show-progress https://wit3.fbk.eu/archive/2016-01//texts/de/en/de-en.tgz | tar xz; mv de-en corpora
```
* STEP 2. Adjust hyper parameters in `hyperparams.py` if necessary.
* STEP 3. Run `prepro.py` to generate vocabulary files to the `preprocessed` folder.
* STEP 4. Run `train.py` or download the [pretrained files](https://www.dropbox.com/s/fo5wqgnbmvalwwq/logdir.zip?dl=0).

## Training Loss and Accuracy
* Training Loss
<img src="fig/mean_loss.png">

* Training Accuracy
<img src="fig/accuracy.png">

## Evaluation
  * Run `eval.py`.