import numpy as np                                                         # NumPy: arrays and math
from step01_softmax import softmax                                         # reuse step 1: scores -> weights
from step02_embeddings import embed, tgt_vocab, tgt_embeddings, src_vocab, src_embeddings  # reuse step 2
from step03_encoder import encode                                          # reuse step 3: encoder states
from step04_decoder import decoder_step                                    # reuse step 4: decoder state
from step05_scoring import additive_scores                                 # reuse step 5: scoring function


def context_vector(weights, H):
    return weights @ H                    # (4,) @ (4,4) -> (4,): weighted sum of the rows of H


if __name__ == "__main__":
    sentence = ["The", "cat", "sat", "today"]
    H = encode(embed(sentence, src_vocab, src_embeddings))                 # encoder states h1..h4, shape (4, 4)

    # Part A: the weights from the running example (what a TRAINED model would give)
    weights = softmax(np.array([0.3, 3.0, 1.4, 0.3]))                      # step 1: ~[0.05, 0.75, 0.15, 0.05]
    parts = weights[:, None] * H                                           # (4,1) * (4,4): scale each h_i by its weight, shape (4, 4)
    context = parts.sum(axis=0)                                            # add the 4 scaled rows together -> shape (4,)

    print("weights        :", weights.round(3))
    print("0.75 * h2 (cat):", parts[1].round(3))                           # the biggest piece of the sum
    print("context        :", context.round(3), "shape", context.shape)
    print("h2 ('cat')     :", H[1].round(3))                               # compare: context is pulled toward this
    print("same via @     :", context_vector(weights, H).round(3))         # one-line version gives the same numbers

    # Part B: the same thing with our untrained scorer from step 5
    s1 = decoder_step(embed(["Le"], tgt_vocab, tgt_embeddings)[0], H[-1])  # decoder state after writing "Le"
    w_untrained = softmax(additive_scores(s1, H))                          # scores -> weights (near-flat, random weights)
    print("untrained weights:", w_untrained.round(3))
    print("untrained context:", context_vector(w_untrained, H).round(3))
