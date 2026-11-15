import numpy as np                                                         # NumPy: arrays and math
from step01_softmax import softmax                                         # reuse step 1: scores -> weights
from step02_embeddings import embed, tgt_vocab, tgt_embeddings, src_vocab, src_embeddings  # reuse step 2
from step03_encoder import encode, HID                                     # reuse step 3: encoder states
from step04_decoder import decoder_step                                    # reuse step 4: decoder state

ATT = 4                                                                    # size of the scoring network's inner layer (real: 256-1024)
rng = np.random.default_rng(3)                                             # new seed for the scoring weights
W1 = rng.normal(size=(ATT, HID)) * 0.5                                     # transforms the decoder state s, shape (4, 4)
W2 = rng.normal(size=(ATT, HID)) * 0.5                                     # transforms each encoder state h, shape (4, 4)
v = rng.normal(size=ATT) * 0.5                                             # final vector that squeezes to ONE number, shape (4,)


def dot_scores(s, H):
    return H @ s                                          # (4,4) @ (4,) -> (4,): dot product of s with every h


def additive_scores(s, H):
    # print("w2", W2)
    inner = np.tanh(W1 @ s + H @ W2.T)    
    # print("inner", inner)
    # W1@s is (4,), H@W2.T is (4,4); broadcasting adds the (4,) to every row
    return inner @ v                                      # (4,4) @ (4,) -> (4,): one score per source word


if __name__ == "__main__":
    sentence = ["The", "cat", "sat", "today"]
    H = encode(embed(sentence, src_vocab, src_embeddings))                 # encoder states, shape (4, 4)
    s0 = H[-1]                                                             # start state from the encoder
    y_le = embed(["Le"], tgt_vocab, tgt_embeddings)[0]                     # vector of "Le"
    s1 = decoder_step(y_le, s0)                                            # decoder state after writing "Le", shape (4,)

    d = dot_scores(s1, H)  
    print("dot scores     :", d.round(3))
    # method A: dot product
    a = additive_scores(s1, H)   
    print("additive scores:", a.round(3))
    # method B: Bahdanau's small network

    # print("dot scores     :", d.round(3), "shape", d.shape)
    # print("dot weights    :", softmax(d).round(3))                         # step 1 used here
    # print("additive scores:", a.round(3), "shape", a.shape)
    print("add. weights   :", softmax(a).round(3))
    # print("target example : [0.3 3.0 1.4 0.3] -> [0.05 0.75 0.15 0.05]  (what a TRAINED model would give)")
