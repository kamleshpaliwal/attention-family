import numpy as np  # NumPy: arrays and math

# Vocabulary: a fixed list of words the model knows. Position in the list = the word's id.
src_vocab = ["The", "cat", "sat", "today"]                  # English (source) words, 4 of them
tgt_vocab = ["Le", "chat", "s'est", "assis", "aujourd'hui"]  # French (target) words, 5 of them

word_to_id = {w: i for i, w in enumerate(src_vocab)}        # dict: word -> id, e.g. "cat" -> 1

EMB_DIM = 4                                                  # size of each word vector (real: 768 or 1536)

rng = np.random.default_rng(0)                               # random generator; seed 0 = same numbers every run
src_embeddings = rng.normal(size=(len(src_vocab), EMB_DIM))  # table of shape (4, 4): one row per English word
tgt_embeddings = rng.normal(size=(len(tgt_vocab), EMB_DIM))  # table of shape (5, 4): one row per French word


def embed(words, vocab, table):
    ids = [vocab.index(w) for w in words]   # turn each word into its id (list of ints)
    return table[ids]                       # fancy indexing: pick those rows -> shape (num_words, EMB_DIM)


if __name__ == "__main__":
    sentence = ["The", "cat", "sat", "today"]
    ids = [word_to_id[w] for w in sentence]
    x = embed(sentence, src_vocab, src_embeddings)  
  


    # print("ids              :", ids)
    # print("src table shape  :", src_embeddings.shape)
    # print("vector for 'cat' :", src_embeddings[word_to_id["cat"]].round(3))
    # print("sentence shape   :", x.shape)
    # print(x.round(3))
    # print("tgt table shape  :", tgt_embeddings.shape)
    # print("vector for 'Le'  :", embed(["Le"], tgt_vocab, tgt_embeddings).round(3), "shape", embed(["Le"], tgt_vocab, tgt_embeddings).shape)
