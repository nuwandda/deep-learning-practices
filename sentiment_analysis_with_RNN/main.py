from data import train_data, test_data
import numpy as np
from model import Model
import random


def main():
    def create_inputs(text):
        # Returns an array of one-hot vectors representing the words in the input text string.
        inputs = []
        for w in text.split(' '):
            v = np.zeros((vocab_size, 1))
            v[word_to_idx[w]] = 1
            inputs.append(v)
        return inputs

    def softmax(xs):
        # Applies the Softmax Function to the input array.
        return np.exp(xs) / sum(np.exp(xs))

    def process_data(data, backprop=True):
        # Returns the RNN's loss and accuracy for the given data.
        items = list(data.items())
        random.shuffle(items)

        loss = 0
        num_correct = 0

        for x, y in items:
            inputs = create_inputs(x)
            target = int(y)

            # Forward
            out, _ = rnn.forward(inputs)
            probs = softmax(out)

            # Calculate loss / accuracy
            loss -= np.log(probs[target])
            num_correct += int(np.argmax(probs) == target)

            if backprop:
                # Build dL/dy
                d_L_d_y = probs
                d_L_d_y[target] -= 1

                # Backward
                rnn.backprop(d_L_d_y)

        return loss / len(data), num_correct / len(data)

    # We have to do some preprocessing before using the data.
    # Create the vocabulary.
    vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
    vocab_size = len(vocab)
    print('%d unique words found' % vocab_size)

    # We need to represent any given word with its corresponding integer index!
    # This is necessary because RNNs can’t understand words
    # Assign indices to each word.
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for i, w in enumerate(vocab)}

    # Initialize our RNN
    rnn = Model(vocab_size, 2)

    # Training loop
    for epoch in range(1000):
        train_loss, train_acc = process_data(train_data)

        if epoch % 100 == 99:
            print('--- Epoch %d' % (epoch + 1))
            print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

            test_loss, test_acc = process_data(test_data, backprop=False)
            print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))

if __name__ == "__main__":
    main()
