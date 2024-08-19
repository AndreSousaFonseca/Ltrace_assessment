# Ltrace_assessment

Here you will find my Ltrace assessment code which revolved around constructing Variational Autoencoders (VAEs). Initially, I constructed a convolutional + dense (fully connected) layer model consisting of 3 convolutional layers and a flat layer and assessed its performance via a loss function. I then implemented a fully convolutional model that contained only 3 convolutional layers. Once again the performance of such a model was assessed. 

Overall, we see that the first model outperforms the second, portrayed by a lower loss after model training ( note: model performance should be also assessed on a test set)

Finally, I wrote a function that recreates images based on the first model.
