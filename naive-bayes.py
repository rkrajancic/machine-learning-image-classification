import numpy as np
import struct
import matplotlib.pyplot as plt

def readFashionMNISTdata():
    # the data files are in the same format (ubyte) as the data files from coding assignment 2 - question2
    # this function was adapted from readMNIST() which was provided in coding assignment 2 
    with open('t10k-images.idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows*ncols))

    with open('t10k-labels.idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size, 1))

    with open('train-images.idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows*ncols))

    with open('train-labels.idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size, 1))

    # randomize other of data
    _random_indices = np.arange(len(train_data))
    np.random.shuffle(_random_indices)
    train_labels = train_labels[_random_indices]
    train_data = train_data[_random_indices]
    
    X_train = train_data[:50000]
    t_train = train_labels[:50000]

    X_val = train_data[50000:]
    t_val = train_labels[50000:]

    return X_train, t_train, X_val, t_val, test_data, test_labels

def class_means_and_variances(X, t, k):
    '''
    estimates the class conditional means and variances for samples belonging to category i
    '''
    N = X.shape[0]
    features = X.shape[1]
    # get all samples that belong to category k
    samples = []
    for i in range(N):
        if (t[i] == k):
            samples.append(i)
    N_samples = len(samples)
    X_aug = np.zeros((N_samples, features))
    for i in range(N_samples):
        X_aug[i] = X[samples[i]]

    # get the mean and varience of each feature from samples belonging to category
    means = np.zeros((features))
    vars = []
    for i in range(features):
        feat = X_aug[:,i]
        means[i] = (np.mean(feat))      # mean
        vars.append(np.var(feat))       # variance
    var_matrix = np.diag(vars)
    # regularize the covarience matrix
    # this is done to prevent having 0s in he digonal
    var_matrix = var_matrix + 1e-6*(np.eye(var_matrix.shape[0]))
    return means, var_matrix  
    
def naive_bayes_predict(X, means, vars, priors, k):
    '''
    use naive bayes theorem to calculate the posterior likelihood
    P(category|x) ∝ P(category)*P(x|category)
    where: 
        P(x|category) ∝e^(-0.5 * (x-μ)^T * Σ^-1 * (x -μ))
    '''
    # precompute the inverses of the covarience matrices to save time:
    for i in range(k):
        vars[i] = np.linalg.inv(vars[i])
    
    N = X.shape[0]
    t_hat = np.zeros((N,1))
    # for each sample in X:
    for i in range(N):
        likelihoods = []
        # for each category
        for j in range(k):
            e = -(1/2) * np.matmul(np.matmul((X[i] - means[j]), vars[j]), (X[i] - means[j]))
            l = priors[j] * np.exp(e)
            likelihoods.append(l)
        category = np.argmax(likelihoods)
        t_hat[i] = category
    return t_hat

def get_accuracy(t, t_hat):
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

def get_priors(t):
    '''
    given a list of labels, calculate the priors for each class
    '''
    N = t.shape[0]
    priors = [0] * 10
    for i in range(N):
        priors[t[i][0]] += 1
    priors = [x / N for x in priors]
    return priors


def main():
    # get data:
    X_train, t_train, X_val, t_val, X_test, t_test = readFashionMNISTdata()

    # in theory, each class is equally likely to occur, so they each have a prior distribution of 1/10
    priors = []
    p = [1/10]*10
    p.append("using equal prior for all categories")
    priors.append(p)
    # however, the data is randomly shuffled, so the priors obtained from the training data may be different
    p = get_priors(t_train)
    p.append("using priors based on training data")
    priors.append(p)

    # estimate class conditional means and variances
    N_category = 10
    means = []
    vars = []
    for i in range(N_category):
        m, v = class_means_and_variances(X_train, t_train, i)
        means.append(m)
        vars.append(v)

    # use validation set to test performance:
    # Hyperparameter tuning:
    #   train using different priors, and then test accuracy of validation data
    #   then use best prior on test data
    val_best = 0
    best_prior = -1
    for i in range(len(priors)):    
        # make predctions:
        t_hat = naive_bayes_predict(X_val, means, vars, priors[i], N_category)
        val_acc = get_accuracy(t_val, t_hat)
        print("Validation accuracy", priors[i][10], ": ", val_acc)
        if (val_acc > val_best):
            val_best = val_acc
            best_prior = i

    # report best prior:
    print("The best prior is:", priors[best_prior][10])
    # test model using testing data and the best prior
    t_hat = naive_bayes_predict(X_test, means, vars, priors[best_prior], N_category)
    acc = get_accuracy(t_test, t_hat)
    print("Test performance (accuracy):", acc, "\n")

main()
    