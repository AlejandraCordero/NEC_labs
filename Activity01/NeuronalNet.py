import numpy as np

class NeuralNet:
    def __init__(self, n_units, epochs=100, lr=0.001, momentum=0.0,
                 activation='tanh', val_pct=0.0, batch_size=32, random_state=None):
        
        # Architecture
        self.n = [int(x) for x in n_units]   
        self.L = len(self.n)                 
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.fact = activation
        self.val_pct = float(val_pct)
        self.batch_size = int(batch_size)

        # Initialize structures
        self.h = [None] * self.L
        self.xi = [None] * self.L
        self.w = [None] * self.L
        self.theta = [None] * self.L
        self.delta = [None] * self.L
        self.d_w_prev = [None] * self.L
        self.d_theta_prev = [None] * self.L

        # Initialize weights
        rng = np.random.RandomState(random_state)
        for l in range(1, self.L):
            in_dim, out_dim = self.n[l-1], self.n[l]
            bound = np.sqrt(2.0/(in_dim + out_dim))
            self.w[l] = rng.normal(0, bound, size=(out_dim, in_dim))
            self.theta[l] = np.zeros((out_dim, 1))
            self.d_w_prev[l] = np.zeros_like(self.w[l])
            self.d_theta_prev[l] = np.zeros_like(self.theta[l])

        self.train_loss_hist = []
        self.val_loss_hist = []

    # ACTIVATION
    def _g(self, h):
        if self.fact == 'sigmoid':
            return 1 / (1 + np.exp(-h))
        if self.fact == 'tanh':
            return np.tanh(h)
        if self.fact == 'relu':
            return np.maximum(0, h)
        if self.fact == 'linear':
            return h
        #raise ValueError("Unknown activation")

    # DERIVATIVE
    def _g_prime(self, out):
        if self.fact == 'sigmoid':
            return out * (1 - out)
        if self.fact == 'tanh':
            return 1 - out**2
        if self.fact == 'relu':
            return (out > 0).astype(float)
        if self.fact == 'linear':
            return np.ones_like(out)

    # FORWARD
    def _forward(self, X):
        self.xi[0] = X
        for l in range(1, self.L):
            h = self.w[l] @ self.xi[l-1] - self.theta[l]
            self.h[l] = h

            if l == self.L-1:
                # output layer uses LINEAR activation for regression
                # h linear activation
                self.xi[l] = h
            else:
                self.xi[l] = self._g(h)

        return self.xi[-1]

    # BACKWARD
    def _compute_deltas(self, Y):
        L = self.L - 1
        out = self.xi[L]

        # Output layer delta for regression (linear activation)
        delta_L = (out - Y)
        self.delta[L] = delta_L

        # Backpropagate through hidden layers
        for l in range(L, 0, -1):
            if l == L:
                continue
            g_prime = self._g_prime(self.xi[l])
            self.delta[l] = g_prime * (self.w[l+1].T @ self.delta[l+1])

        return self.delta

    # UPDATE WEIGHTS
    def _update_weights(self, batch_X, batch_y):
        X = batch_X.T
        Y = batch_y.reshape(1, -1)

        self._forward(X)
        self._compute_deltas(Y)

        batch_size = X.shape[1]

        for l in range(1, self.L):
            grad_w = self.delta[l] @ self.xi[l-1].T
            grad_theta = np.sum(self.delta[l], axis=1, keepdims=True)

            dw = -self.lr * grad_w + self.momentum * self.d_w_prev[l]
            dtheta = -self.lr * grad_theta + self.momentum * self.d_theta_prev[l]

            self.w[l] += dw / batch_size
            self.theta[l] += dtheta / batch_size

            self.d_w_prev[l] = dw
            self.d_theta_prev[l] = dtheta

    # TRAIN
    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n_samples = len(X)

        # Split
        if self.val_pct > 0:
            idx = np.random.permutation(n_samples)
            cut = int((1 - self.val_pct) * n_samples)
            train_idx, val_idx = idx[:cut], idx[cut:]
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        for epoch in range(self.epochs):
            perm = np.random.permutation(len(X_train))

            for i in range(0, len(X_train), self.batch_size):
                batch = perm[i:i+self.batch_size]
                self._update_weights(X_train[batch], y_train[batch])

            # Compute losses
            y_pred = self.predict(X_train)
            train_loss = np.mean((y_pred - y_train)**2)
            self.train_loss_hist.append(train_loss)

            if X_val is not None:
                y_val_pred = self.predict(X_val)
                val_loss = np.mean((y_val_pred - y_val)**2)
                self.val_loss_hist.append(val_loss)
            else:
                self.val_loss_hist.append(np.nan)

    # PREDICT
    def predict(self, X):
        X = np.asarray(X, float).T
        A = X
        for l in range(1, self.L):
            H = self.w[l] @ A - self.theta[l]
            if l == self.L-1:
                A = H  # linear output
            else:
                A = self._g(H)
        return A.T.ravel()

    # RETURN LOSS CURVES
    def loss_epochs(self):
        return np.column_stack([self.train_loss_hist, self.val_loss_hist])
