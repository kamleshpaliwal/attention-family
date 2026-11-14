import numpy as np  # NumPy: library for arrays (lists of numbers with math)


def softmax(scores):
    exps = np.exp(scores)         # e^x for every score at once (element-wise)
    return exps / exps.sum()      # divide each by the total so everything sums to 1


if __name__ == "__main__":        # only runs when you execute this file directly, not on import
    words = ["The", "cat", "sat", "today"]
    scores = np.array([0.3, 3.0, 1.4, 0.3])   # one score per English word

    exps = np.exp(scores)
    weights = softmax(scores)

    print("words   :", words)
    print("scores  :", scores, "shape", scores.shape)
    print("weights :", weights.round(4), "shape", weights.shape)
    print("sum of weights:", weights.sum())
