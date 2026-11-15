import numpy as np                                                         # NumPy: arrays and math
from step02_embeddings import embed, tgt_vocab, tgt_embeddings, src_vocab, src_embeddings  # reuse step 2
from step03_encoder import encode, HID                                     # reuse step 3: the encoder

rng = np.random.default_rng(2)                                             # new seed, separate from the encoder's weights
Wy = rng.normal(size=(HID, tgt_embeddings.shape[1])) * 0.5                 # weights for the previous French word's vector, (4, 4)
Ws = rng.normal(size=(HID, HID)) * 0.5                                     # weights for the previous decoder state, (4, 4)
b = np.zeros(HID)                                                          # bias, shape (4,)


def decoder_step(y_prev, s_prev):
    return np.tanh(Wy @ y_prev + Ws @ s_prev + b)   # same recipe as the encoder RNN, but fed with the last WRITTEN French word


if __name__ == "__main__":
    sentence = ["The", "cat", "sat", "today"]
    H = encode(embed(sentence, src_vocab, src_embeddings)) 
    print("hhhh",H)# encoder states h1..h4, shape (4, 4)

    s0 = H[-1] # start state: the encoder's last state (its summary of the sentence)
    y_le = embed(["Le"], tgt_vocab, tgt_embeddings)[0]                     # vector of the French word "Le", shape (4,)
    s1 = decoder_step(y_le, s0)                                            # after writing "Le" -> s1; used to pick "chat"

    y_chat = embed(["chat"], tgt_vocab, tgt_embeddings)[0]                 # vector of "chat", shape (4,)
    s2 = decoder_step(y_chat, s1)                                          # after writing "chat" -> s2; used to pick the next word

    # print("s0 (from encoder):", s0.round(3), "shape", s0.shape)
    # print("y 'Le'           :", y_le.round(3))
    # print("s1               :", s1.round(3), "shape", s1.shape)
    # print("y 'chat'         :", y_chat.round(3))
    # print("s2               :", s2.round(3), "shape", s2.shape)
