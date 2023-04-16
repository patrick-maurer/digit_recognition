import numpy as np
import modules.helper_functions as hf


class NeuralNetwork:
    """
    My first neural network
    """

    def __init__(self, Ws=None, y_classes=None):
        """
        initialization

        :param Ws: optional list of weight matrices (list of 2-D numpy arrays)
        :param y_classes: optional array of y_classes (1-D numpy array with 2 elements)
        """
        self.Ws = Ws
        self.y_classes = y_classes

    def fit(
        self, X, y, hiddenNodes, stepSize=0.01, ITERS=100, batchSize=None, seed=None
    ):
        """
        Find the best weights via stochastic gradient descent

        :param X: training features
        :param y: training labels
        :param hiddenNodes: list of hidden nodes in each layer (excluding bias nodes)
        :param stepSize: learning rate
        :param ITERS: number of epochs
        :return: update self.y_classes, self.Ws
        """

        # validate X dimensionality
        if X.ndim != 2:
            raise AssertionError(f"X should have 2 dimensions but it has {X.ndim}")

        if not isinstance(hiddenNodes, list):
            raise AssertionError("hiddenNodes schould be a list of integers")

        y01, y_classes = hf.one_hot(y)
        if len(y_classes) < 2:
            raise AssertionError(f"y should have at least 2 distinct classes")

        # initialization
        generator = np.random.default_rng(seed)  # rng to create batches

        X1 = np.hstack((X / 255, np.ones(shape=(X.shape[0], 1))))  # normalize values

        Ws = [None] * (len(hiddenNodes) + 1)
        Ws[0] = generator.uniform(low=-1, high=1, size=(X1.shape[1], hiddenNodes[0]))
        for i in range(1, len(hiddenNodes)):
            Ws[i] = generator.uniform(
                low=-1, high=1, size=(hiddenNodes[i - 1] + 1, hiddenNodes[i])
            )
        Ws[i + 1] = generator.uniform(
            low=-1, high=1, size=(hiddenNodes[i] + 1, len(y_classes))
        )

        # initialize lists to store Xs, Zs, and gradients
        Zs = [None] * len(Ws)
        Xs = [None] * len(Ws)
        gradWs = [None] * len(Ws)

        # determine number of bathces
        if batchSize is None:
            Nbatches = 1
        else:
            Nbatches = np.ceil(X1.shape[0] / batchSize).astype("int64")

        for i in range(ITERS):
            # create mini batches
            idxs = generator.choice(X1.shape[0], size=X1.shape[0], replace=False)
            batches = np.array_split(idxs, Nbatches)

            for b in range(Nbatches):
                batch_idxs = batches[b]
                Xs[0] = X1[batch_idxs]

                # forward pass
                for j in range(len(Ws)):
                    Zs[j] = Xs[j] @ Ws[j]
                    if j + 1 < len(Xs):
                        Xs[j + 1] = np.hstack(
                            (hf.logistic(Zs[j]), np.ones(shape=(Zs[j].shape[0], 1)))
                        )
                yhat_probs = hf.softmax(Zs[-1])
                yhat_classes = y_classes[np.argmax(yhat_probs, axis=1)]

                # calculate cross entropy loss and accuracy
                ce = hf.cross_entropy(yhat_probs, y01[batch_idxs])
                CE = np.mean(ce)
                accuracy = np.mean(yhat_classes == y[batch_idxs])

                if b == 0:
                    print(
                        f"iteration: {i}, batch: {b}, cross entropy loss: {CE}, accuracy: {accuracy}"
                    )

                # calculate gradients (backward pass)
                gradZ = (yhat_probs - y01[batch_idxs])[:, None, :]
                for j in range(len(Ws) - 1, -1, -1):
                    gradWs[j] = np.transpose(Xs[j][:, None, :], axes=[0, 2, 1]) @ gradZ
                    gradWs[j] = gradWs[j].mean(axis=0)
                    gradX = (gradZ @ np.transpose(Ws[j]))[:, :, :-1]
                    gradZ = gradX * (Xs[j] * (1 - Xs[j]))[:, None, :-1]

                # update weights (gradient step)
                for j in range(len(Ws)):
                    Ws[j] -= gradWs[j] * stepSize

        # update class attributes
        self.Ws = Ws
        self.y_classes = y_classes

    def predict(self, X, type="classes"):
        """
        predict on X

        :param X: 2-D array with columns of features
        :return: 1-D array of predicted values
        """

        if self.Ws is None:
            raise AssertionError(f"Need to fit() before predict()")
        if X.ndim != 2:
            raise AssertionError(f"X should have 2 dimensions but it has {X.ndim}")
        if X.shape[1] != len(self.Ws[0]) - 1:
            raise AssertionError(
                f"Perceptron was fit on X with {len(self.Ws[0]) - 1} columns but this X has {X.shape[1]} columns"
            )

        # Make predictions (forward pass)
        X1 = np.insert(X / 255, obj=X.shape[1], values=1, axis=1)
        for j in range(len(self.Ws)):
            Z = X1 @ self.Ws[j]
            if j < len(self.Ws) - 1:
                X1 = np.insert(hf.logistic(Z), obj=Z.shape[1], values=1, axis=1)
        yhat_probs = hf.softmax(Z)

        if type == "probs":
            return yhat_probs
        elif type == "classes":
            yhat_classes = self.y_classes[np.argmax(yhat_probs, axis=1)]
            return yhat_classes
