
### notes
This doesn't match the tensorflow reference implementation very well. For a keras implementation of mobilenet see https://keras.io/applications/#mobilenet

### mobilenet trained on imagenet
Runs with tensorflow 1.1 does not require keras as it uses the keras api now included in tensorflow contrib. Trained rgb images rescaled to 256x256(no cropping) using ((x/255)-0.5) \* 2 to scale pixel values. It gets 0.6 top one accuracy on the full imagenet cls validation set(did not remove blacklist). Validation error was still decreaseing when I stopped training. I may update weights if I have time. 

Link to the mobilenet paper. https://arxiv.org/abs/1704.04861
