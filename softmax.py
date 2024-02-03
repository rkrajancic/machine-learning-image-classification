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

    # augmenting a constant feature of 1 (absorbing the bias term)
    train_data = np.concatenate(
        (np.ones([train_data.shape[0], 1]), train_data), axis=1)
    test_data = np.concatenate(
        (np.ones([test_data.shape[0], 1]),  test_data), axis=1)
    _random_indices = np.arange(len(train_data))
    np.random.shuffle(_random_indices)
    train_labels = train_labels[_random_indices]
    train_data = train_data[_random_indices]
    
    X_train = train_data[:50000]
    t_train = train_labels[:50000]

    X_val = train_data[50000:]
    t_val = train_labels[50000:]

    return X_train, t_train, X_val, t_val, test_data, test_labels
    
def predictSoftmax(X, W, t, N_class, get_accuracy):
    # this function was adapted from my submission for question 2 of coding assignment 2 
    # X_new: Nsample x (d+1)
    # W: (d+1) x K

    # get z vectors
    N = X.shape[0]                  # N is the number of samples being dealt with
    z = np.zeros((N, N_class))
    y = np.zeros((N, N_class))
    for i in range(N):
        # for each sample in X
        # get it z vector
        z[i] = np.matmul(X[i], W)
        # for each zi in z subtract every zi by the maximum value to prevent overflow
        max_z = z[i].max()
        for j in range(N_class):
            z[i,j] -= max_z
        
        # Softmax regression:
        # get the y vector with prediction probabilities
        y[i] = np.exp(z[i])
        sum = 0
        for j in range(N_class):
            sum += np.exp(z[i][j])
        y[i] = y[i] / sum
    
    # loss:
    # loss = J(w) =  - sum( (ti - log(yi))
    # ti is a one-hot vecot for the category corresponding to ti
    loss = 0
    for i in range(N):
        # for each sample in X
        # one hot encoding for each t
        one_hot = t[i]
        if (y[i][one_hot] == 0):
            J_i = np.log(1e-16)
        else:
            J_i = np.log(y[i][one_hot])
        loss -= J_i
    loss *= (1/N)

    if (get_accuracy):
        # predictions t_hat:
        t_hat = np.zeros((N))
        for i in range(N):
            t_hat[i] = np.argmax(y[i])
        # accuracy:
        acc = 0
        for i in range(N):
            if (t_hat[i] == t[i]):
                acc += 1
        acc = acc / N
        return acc
    else:
        return y, loss

def train(X_train, y_train, X_val, t_val, N_class, batch_size, MaxEpoch, alpha):
    # this function was adapted from my submission for question 2 of coding assignment 2 
    N_train = X_train.shape[0]          # number of training samples
    
    # initialize W matrix (785x10)
    W = np.random.rand(X_train.shape[1], N_class)
    # W = np.random.uniform(-0.5, 0.5, (X_train.shape[1], N_class))
    # initialize other values
    train_losses = []
    valid_accs = []
    acc_best = 0
    W_best = None
    epoch_best = 0

    for epoch in range(MaxEpoch):              
        loss_this_epoch = 0
        for b in range(int(np.ceil(N_train/batch_size))):
            X_batch = X_train[b*batch_size: (b+1)*batch_size]
            y_batch = y_train[b*batch_size: (b+1)*batch_size]
            
            y, loss_batch = predictSoftmax(X_batch, W, y_batch, N_class, False)
            loss_this_epoch += loss_batch
            
            # calculate the gradient:
            # make a one-hot encoding of y_batch
            one_hot_true_value = np.zeros((batch_size, 10))
            for i in range(batch_size):
                category = y_batch[i]
                one_hot_true_value[i,category] = 1
            
            temp = y - one_hot_true_value
            J_W = np.matmul(np.transpose(X_batch), temp)
            J_W = J_W / batch_size
            W = W - alpha*J_W
        
        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        num_of_iterations = int(np.ceil(N_train/batch_size))
        training_loss = loss_this_epoch / num_of_iterations
        train_losses.append(training_loss)
        
        
        # 2. Perform validation on the validation set by the risk
        acc = predictSoftmax(X_val, W, t_val, N_class, True)
        valid_accs.append(acc)

        # 3. Keep track of the best validation epoch, risk, and the weights
        if (acc > acc_best):
            acc_best = acc
            epoch_best = epoch
            W_best = W
           
    # Return some variables as needed
    return epoch_best, acc_best,  W_best, train_losses, valid_accs

def main():
    # get the data form input files
    # there are 50,000 training sample
    # 10,000 validation samples
    # 10,000 test smples
    X_train, t_train, X_val, t_val, X_test, t_test = readFashionMNISTdata()
    # there are 10 classes to consider for this classification problem
    N_class = 10
    MaxEpoch = 50                                  
    # Hyperparameter tuning:
    # systematically tune alpha from list of preselected values
    # train model with each alpha
    # evaluate model performance on validation set
    # select the alpha value that gives best results and then use it on training data
    alpha =[ 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
    acc_best = 0
    alpha_best = None
    W_best = None
    for i in range(len(alpha)):
        epoch_best, acc,  W, train_losses, valid_accs = train(X_train, t_train, X_val, t_val, N_class, 100, MaxEpoch, alpha[i])
        if (acc > acc_best):
            acc_best = acc
            alpha_best = alpha[i]
            W_best = W
        print("Using learning rate: ", alpha[i])
        print("Epoch that yields the best validation performance: ", epoch_best)
        print("Validation performance (accuracy) in that epoch: ", acc)
        print()
    
    # report the best alpha value
    print("Best alpha value based on hyperparameter tuning: ", alpha_best)

    # test model using testing data and W obtained using best alpha value
    acc_test = predictSoftmax(X_test, W_best, t_test, N_class, True)
    print("Test performance (accuracy) on testing data: ", acc_test)

    '''
    # plot learning curve of the training loss
    e = list(range(MaxEpoch))
    plt.plot(e, train_losses)
    plt.title("SOFTMAX: Learning Curve of the Training Cross-Entropy Loss")
    plt.xlabel('number of epochs')
    plt.ylabel('training loss')
    plt.tight_layout()
    plt.savefig('SOFTMAX_learning_curve_of_training_loss.jpg')

    # plot learning curve of validation risk
    plt.figure()
    plt.plot(e, valid_accs)
    plt.title("SOFTMAX: Learning Curve of Validation Accuracy")
    plt.xlabel('number of epochs')
    plt.ylabel('validation risk')
    plt.tight_layout()
    plt.savefig('SOFTMAX_learning_curve_of_validation_accuracy.jpg')
    '''

main()


