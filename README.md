# 546 Final Project: Masked Autoencoder
##### Haoran Tang, Qirui Wu

#### 1. Training
To train the network, please run mae_pretraining.py. Please modify folder paths and args if necessary. Also, to modify embedding dimension of the encoder, please change it manually in models.py, the VIT network.
### 2. Results
We save training loss curves in the checkpoint, and to visualize the losses please load from checkpoint (only a list is needed) in draw.py. PLease modify paths if necessary. We also record the test accuracy at last epoch, but we report the best of the last 10 epochs.
### 3. Other files
Models are defined in models.py, datasets in datasets.py, KNN test in knntest.py, initialization of optimizers in utils.py, 
### 4. Log files
We saved the last epoch for each experiment (total 12 checkpoints), most of then are large. If you need a checkpoint to test please let us know, thank you!
