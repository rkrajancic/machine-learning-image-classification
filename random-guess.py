import numpy as np
import struct
import matplotlib.pyplot as plt
import random

def readFashionMNISTdata_targets():
    # the data files are in the same format (ubyte) as the data files from coding assignment 2 - question2
    # this function was adapted from readMNIST() which was provided in coding assignment 2
    # for random guessing, we only need the target values

    with open('t10k-labels.idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size, 1))
   
    return test_labels

def random_guess(N,K):
    '''
    creates an array of N random guesses from K categories
    '''
    guesses = np.zeros((N,1))
    for i in range(N):
        guesses[i] = random.randint(0,K-1)
    return guesses

def get_accuracy(t_hat, t):
    '''
    gets the accuracy of predictions form t_hat
    achieved by counting the number of correct predicitons and then dividing by the number of samples
    '''
    N = t.shape[0]
    acc = 0
    for i in range(N):
        if (t[i] == t_hat[i]):
            acc += 1
    acc /= N
    return acc

def main():
    # read the labels from the dataset
    test_labels = readFashionMNISTdata_targets()
    guess = random_guess(test_labels.shape[0], 10)    
    acc = get_accuracy(guess, test_labels)
    print("Accuracy of random guesses: ", acc)
    return

main()