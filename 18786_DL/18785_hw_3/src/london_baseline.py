# TODO: [part d]
# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.

import argparse
import utils

def main():
    accuracy = 0.0

    # Compute accuracy in the range [0.0, 100.0]
    ### YOUR CODE HERE ###

    # create prediction set.
    predicted_vals = []
    with open("birth_dev.tsv",'r', encoding="utf-8") as input_set: 
        for line_i in input_set:
            if (line_i):
                predicted_vals.append("London")

    # use evaluate places from utils to find num total and num correct. 
    total, correct = utils.evaluate_places("birth_dev.tsv", predicted_places=predicted_vals)
    accuracy = (correct / total) * 100

    ### END YOUR CODE ###

    return accuracy

if __name__ == '__main__':
    accuracy = main()
    with open("london_baseline_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy}\n")
