> *seq2seq* models.

* Notes:

    - Neural Machine Translation (NMT) is one of the first testbeds of *seq2seq* models.

    - NMT mimics the behaviour of understanding (encoding) a given sentence and then translating (decoding). 
        In another word a **Encoder-Decoder** architecture.

    - Recurrent Neural Network (RNN) is a natural choice for sequential data, used by most NMT models. However it 
        could be uni or bidirectional, single or multi-layer, vanilla or a Long Short-term Memory (LSTM) RNN.

    - This note/tutorial which is based on tensorflow's nmt repository (reference #1), uses a deep multi-layer RNN 
        which is unidirectional and uses LSTM as a recurrent unit. At a high level, the NMT model consists of two 
        recurrent neural networks: the *encoder* RNN simply consumes the input source words without making any prediction; 
        the *decoder* processes the target sentence while predicting the next words.
            <img src="assets/seq2seq.jpg" style="display:block;margin-left:auto; margin-right:auto; width:75%">

    - We try in this part to give some notes and summary of the [model.py](https://github.com/tensorflow/nmt/blob/master/nmt/model.py) 
        file of the tensorflow's nmt repository.

    - At the bottom layer, the encoder and decoder RNNs receive as input the following: 
        first, the source sentence, then a boundary marker "\<s\>" which indicates the 
        transition from the encoding to the decoding mode, and the target sentence.

    - For training, the following word indices tensors are used:
        - *encoder_inputs*: [max_encoder_time, batch_size]: source input words.
        - *decoder_inputs*: [max_decoder_time, batch_size]: target input words.
        - *decoder_outputs*: [max_decoder_time, batch_size]: target output words, 
            these are decoder_inputs shifted to the left by one time step 
            with an end-of-sentence tag appended on the right.

    - Forthub

---
* Resources:
    1. [Tensorflow NMT](https://github.com/tensorflow/nmt)
    2. [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

