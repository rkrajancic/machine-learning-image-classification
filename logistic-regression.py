import numpy as np
import struct
import matplotlib.pyplot as plt
# Used for testing:
from sklearn.linear_model import LogisticRegression



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

    # standardize the data the data:
    mean_vals = np.mean(train_data, axis=0)
    std_devs = np.std(train_data, axis=0)
    train_data = (train_data - mean_vals) / std_devs
    mean_vals = np.mean(test_data, axis=0)
    std_devs = np.std(test_data, axis=0)
    test_data = (test_data - mean_vals) / std_devs

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

def predict_logistic_regression(X, W):
    '''
    given matrix X with N samples, use the weights to make a prediction
    for each sample the function will produce 10 numbers
    these numbers represent the probability of the sample belongig to each of the 10 categories
    pick the category with the highest probability as your prediction
    '''
    N = X.shape[0]
    t_hat = []
    t_hat_pre_classification = np.matmul(X,W)
    for i in range(N):
        category = np.argmax(t_hat_pre_classification[i])
        t_hat.append(category)
    return t_hat

def train_logistic_regression(X_train, t_train, X_val, t_val, N_class, batch_size, MaxEpoch, alpha):
    """
    Given data, train your logistic classifier.
    """
    N = X_train.shape[0]        # training with N samples
    # Need 10 seperate W vectors.
    # Each logistic classifier gets ites own W vector

    W = np.random.rand(X_train.shape[1], N_class)
    W_best = None
    acc_best = 0
    epoch_best = 0
    train_losses = []
    valid_accs = []
    # NOTE: training 10 different logistic regression models
    # one model per class
    for epoch in range(MaxEpoch):      
        loss_this_epoch = 0
        for batch in range(int(np.ceil(X_train.shape[0]/batch_size))):
            X_batch = X_train[batch*batch_size: (batch+1)*batch_size]
            # get t_batch for each logistic classifier
            # t_train is a list of 10 sets of training data (one for each model)
            t_batch= []
            for i in range(N_class):
                t_batch_class_i = t_train[i][batch*batch_size: (batch+1)*batch_size]
                t_batch.append(t_batch_class_i)
            
                
            predictions = sigmoid(X_batch, W)
            
            l = loss(predictions, t_batch, N_class)
            loss_this_epoch += l
            t_batch = np.array(t_batch)
            t_batch = np.transpose(t_batch)
          
            # Mini-batch gradient descent
            J_w = gradient(predictions, t_batch, X_batch)
            W = W -alpha*J_w
         
        
        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        num_of_iterations = int(np.ceil(X_train.shape[0]/batch_size))
        training_loss = loss_this_epoch / num_of_iterations
        train_losses.append(training_loss)

        # 2. Perform validation on the validation set by the risk
        # can get accuracy as a measure of validation
        t_hat = predict_logistic_regression(X_val, W)
        acc = get_accuracy(t_val, t_hat)
        valid_accs.append(acc)

        # 3. Keep track of the best validation epoch, risk, and the weights
        if (acc >= acc_best):
            acc_best = acc
            epoch_best = epoch
            W_best = W 

    return epoch_best, acc_best,  W_best, train_losses, valid_accs    

def sigmoid(X,W):
    '''
    calculate the sigmoid functions for given X, and Ws
    X is a matrix with N samples
    '''
    N = X.shape[0]
    z = np.zeros((N, 10))
    for i in range(N):
        z[i] = np.matmul(X[i], W)
    z = np.clip(z, -600, None)      # used to prevent overflow at runtime 
    y = 1 / (1+np.exp(-z))
    return y

def loss(t_hat, t, N_class):
    '''
    calculate the loss for logistic regression
    J(w,b) = (1/M) * sum {-t_m*log(y_m) - (1-t_m)*log(1-y_m)
    '''
    N = t_hat.shape[0]
    total_loss = 0
    for cat in range(N_class):
        loss = 0
        for i in range(N):
            if (t_hat[i][cat] <= 1e-16):
                J_m = -t[cat][i]*(np.log(1e-16)) - (1-t[cat][i])*(np.log(1-1e-16))
            elif (t_hat[i][cat] == 1):
                J_m = -t[cat][i]*(np.log(t_hat[i][cat])) - (1-t[cat][i])*(np.log(1e-16))
            else:
                J_m = -t[cat][i]*(np.log(t_hat[i][cat])) - (1-t[cat][i])*(np.log(1-t_hat[i][cat]))
            loss += J_m
        loss *= (1/N)   
        total_loss += loss     
    return total_loss

def gradient(predictions, t, X):
    '''
    gradient -> 1/N * X^T (predictions - targets)
    '''
    N = X.shape[0]
    J_w = np.transpose(X)
    diff = predictions - t
    J_w = np.matmul(J_w, diff)
    return J_w

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


def main():
    # get the data form input files
    # there are 50,000 training sample
    # 10,000 validation samples
    # 10,000 test smples
    X_train, t_train, X_val, t_val, X_test, t_test = readFashionMNISTdata()
    # there are 10 classes to consider for this classification problem
    N_class = 10
    MaxEpoch = 50                                  

    # Problem deals with multiple classes:
    # Need to use a One-vs-Rest (OvR) Approach to Logistic Regression
    # For each class train a binary logistic regression classifier
    # each classifier will focus on seperating one class from the rest
    # for class i: 
    #   samples in category i will have target value of 1
    #   all other samples will have target values of 0
    
    # setup the modified training targets
    t_train_one_vs_all = []
    for i in range(N_class):
        t_train_i = []
        for x in range(t_train.size):
            if (t_train[x] == i):
                t_train_i.append(1)
            else:
                t_train_i.append(0)
        t_train_one_vs_all.append(t_train_i)

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
        epoch_best, acc,  W, train_losses, valid_accs = train_logistic_regression(X_train, t_train_one_vs_all, X_val, t_val, N_class, 100, MaxEpoch, alpha[i])
        if (acc > acc_best):
            acc_best = acc
            W_best = W
            alpha_best = alpha[i]
        print("Using learning rate: ", alpha[i])
        print("Epoch that yields the best validation performance: ", epoch_best)
        print("Validation performance (accuracy) in that epoch: ", acc)
        print()

    # report the best alpha value
    print("Best alpha value based on hyperparameter tuning: ", alpha_best)

    # test model using testing data and W obtained using best alpha value
    test_predictions = predict_logistic_regression(X_test, W_best)
    acc_test = get_accuracy(t_test, test_predictions)
    print("Test performance (accuracy) on testing data: ", acc_test)

    '''
    # plot learning curve of the training loss
    e = list(range(MaxEpoch))
    plt.plot(e, train_losses)
    plt.title("LOGISTIC-REGRESSION: Learning Curve of the Training Cross-Entropy Loss")
    plt.xlabel('number of epochs')
    plt.ylabel('training loss')
    plt.tight_layout()
    plt.savefig('LR_learning_curve_of_training_loss.jpg')

    # plot learning curve of validation risk
    plt.figure()
    plt.plot(e, valid_accs)
    plt.title("LOGISTIC_REGRESSION: Learning Curve of Validation Accuracy")
    plt.xlabel('number of epochs')
    plt.ylabel('validation risk')
    plt.tight_layout()
    # plt.ylim(0,1)
    plt.savefig('LR_learning_curve_of_validation_accuracy.jpg')
    '''

main()