# Final-Project
This is Linda Ahlstrom's Final Project with VAE combined w/ RL
Here is my final project    LINDA AHLSTROM

the git hub link is https://github.com/Linda4u/Linda4u.git

you will find 4 final project files there under Final Project Report Fall 2025

VAEGARCHver3origional.py
sp500.py
sp500c.csv
vae_keras_dense_predict.keras


The first file is the main file VAE
The second file loads the data and is called by the VAE file
the .csv file contains the data
the output file generated and saved as the model is vae_keras_dense_predict.keras  this file is saved after training and loaded for testing

there is a place in VAE to set Train = True for training that has to be manually switched to False for testing   There is a create random noise function in sp500.py that can by manually turned on and off.  This is for the additional option to add Robustness evaluation through domain randomization or noise injection.
