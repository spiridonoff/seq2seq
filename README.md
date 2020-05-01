# seq2seq
A model to combine two sub-captions for image captioning on VizWiz dataset

split_captions.ipynb is used to create the dataset used for seq-2-seq model.
eval.ipynb is used to evaluate the trained model on the validation/test dataset and create captions from smaller subcaptions.
opts.py gathers the main parameters of the network to be imported by other python codes.
prepre_labels.py is used to preprocess the captions and create a word embedding /language model. This file generates the language.pkl which can be imported by other modules of the project.
read_tsv.py is used to read the tsv file containing the bottom-up features and store them in npz files.
train.py is the main file used to train the network.
utils.py contains some functions used in the project.

For final evaluation of generated captions, we use the API presented by VizWiz at 
https://github.com/Yinan-Zhao/vizwiz-caption
