"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    num_dim = W.shape[0]
    num_cls = W.shape[1]
    num_tr = X.shape[0]
    
    for i in range(num_tr):
        f = X[i,:].dot(W)
        exp_f = np.exp(f)
        p = exp_f / np.sum(exp_f)
        loss += -np.log(p[y[i]])
        for k in range(num_dim):
            for l in range(num_cls):
                if l == y[i]:
                    dW[k,l] += X.T[k,i]*(p[l]-1)
                else:
                    dW[k,l] += X.T[k,i]*p[l]
    loss /= num_tr
    loss += 0.5 * reg * np.sum(W**2)
    
    dW /= num_tr
    dW += reg * W
        
        
    
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    num_tr = X.shape[0]
    f = np.dot(X,W)
    f = f - np.max(f,axis = 1,keepdims = 1)
    exp_f = np.exp(f)
    p = exp_f / np.sum(exp_f,axis = 1, keepdims = 1)
    loss += -np.log(p[range(num_tr),y])
    loss = np.sum(loss)
    loss /= num_tr
    loss += 0.5 * reg * np.sum(W**2)
    
    df = p
    df[range(num_tr),y] -= 1
    df /= num_tr
    dW = np.dot(X.T,df)
    dW +=  reg * W
        
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-7, 5e-7]
    regularization_strengths = [1e4, 1.5e4]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################
    for lr in learning_rates:
        for reg in regularization_strengths:
            softmax = SoftmaxClassifier()
            softmax.train(X_train, y_train, learning_rate=lr, reg=reg,
                          num_iters= 1000, verbose=True)
            y_train_pred = softmax.predict(X_train)
            y_val_pred = softmax.predict(X_val)
            train_accurary = np.mean(y_train == y_train_pred)
            val_accurary = np.mean(y_val == y_val_pred)
            if val_accurary > best_val:
                best_softmax = softmax
                best_val = val_accurary
            all_classifiers.append([softmax,val_accurary])
            results[(lr,reg)] = train_accurary, val_accurary
           
    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
