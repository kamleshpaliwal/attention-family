import numpy as np                                            # NumPy: arrays and math
from step02_embeddings import embed, src_vocab, src_embeddings, EMB_DIM   # reuse step 2: word -> vector

HID = 4                                                        # size of each encoder state (real: 512-1024)

rng = np.random.default_rng(1)                                 # seeded generator -> same numbers every run
Wx = rng.normal(size=(HID, EMB_DIM)) * 0.5                     # weights applied to the current word vector, shape (4, 4)
Wh = rng.normal(size=(HID, HID)) * 0.5                         # weights applied to the previous state, shape (4, 4)
b = np.zeros(HID)                                              # bias: a constant added each step, shape (4,)


def rnn_step(x, h_prev):
    return np.tanh(Wx @ x + Wh @ h_prev + b)    # @ = matrix times vector; tanh squashes every value into (-1, 1)


def encode(x_seq):
    # print("x_seq", x_seq)
    h = np.zeros(HID)                           # start state: all zeros ("nothing read yet")
    states = []                                 # will collect h1..h4
    for x in x_seq:  
        # print("x", x)
        # go through the words left to right
        h = rnn_step(x, h)                      # new state mixes this word with everything read so far
        states.append(h)                        # keep it
    return np.array(states)                     # stack into one array, shape (num_words, HID)


if __name__ == "__main__":
    sentence = ["The", "cat", "sat", "today"]
    x_seq = embed(sentence, src_vocab, src_embeddings)         # (4, 4): the word vectors from step 2

    H_given = x_seq                                            # Part A: simplest "encoder" = use the vectors as they are
    # print("A) given states, shape", H_given.shape)
    # print(H_given.round(3))

    H = encode(x_seq)                                          # Part B: states produced by the RNN
    # print("B) RNN states, shape", H.shape)
    # print(H.round(3))
    # print("h2 ('cat') :", H[1].round(3))
