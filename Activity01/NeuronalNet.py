import numpy as np

class NeuralNet:
    def __init__(self, n_units, epochs=100, lr=0.01, momentum=0.0,
                 activation='tanh', val_pct=0.0, batch_size=1, random_state=None):
        # n_units: list like [n_input, hidden1, ..., n_output]
        self.n = [int(x) for x in n_units]   # n
        self.L = len(self.n)                 # L
        self.epochs = int(epochs)
        self.lr = float(lr)                  # learning rate (eta)
        self.momentum = float(momentum)      # alpha
        self.fact = activation               # activation name
        self.val_pct = float(val_pct)        # validation percentage
        self.batch_size = int(batch_size)

        # Required arrays and structures (index 0 unused for weights idea)
        self.h = [None] * self.L
        self.xi = [None] * self.L
        self.w = [None] * self.L
        self.theta = [None] * self.L
        self.delta = [None] * self.L
        self.d_w = [None] * self.L
        self.d_theta = [None] * self.L
        self.d_w_prev = [None] * self.L
        self.d_theta_prev = [None] * self.L

        # Initialize weights and thresholds (biases)
        rng = np.random.RandomState(random_state)
        for l in range(1, self.L):
            in_dim, out_dim = self.n[l-1], self.n[l]
            bound = np.sqrt(2.0/(in_dim + out_dim)) 
            self.w[l] = rng.normal(0, bound, size=(out_dim, in_dim))
            self.theta[l] = np.zeros((out_dim, 1))
            self.d_w_prev[l] = np.zeros_like(self.w[l])
            self.d_theta_prev[l] = np.zeros_like(self.theta[l])

        # history
        self.train_loss_hist = []
        self.val_loss_hist = []

    # 2. Activation and derivate function
    def _g(self, h):
        if self.fact == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-h))
        if self.fact == 'tanh':
            return np.tanh(h)
        if self.fact == 'relu':
            return np.maximum(0, h)
        if self.fact == 'linear':
            return h
        raise ValueError("Unknown activation")

    def _g_prime(self, h, out=None):
        if self.fact == 'sigmoid': 
            s = out if out is not None else 1.0/(1.0+np.exp(-h))
            return s * (1 - s)
        if self.fact == 'tanh':
            t = out if out is not None else np.tanh(h)
            return 1 - t**2
        if self.fact == 'relu':
            return (h > 0).astype(float)
        if self.fact == 'linear':
            return np.ones_like(h)
        
    # 3. Forward pass
    def _forward(self, X_mat):
        # When there is X_mat should be shape with n_features and n_samples)
        self.xi[0] = X_mat
        for l in range(1, self.L):
            h_l = self.w[l].dot(self.xi[l-1]) - self.theta[l]
            self.h[l] = h_l
            self.xi[l] = self._g(h_l)
        return self.xi[self.L-1]   # activations
    
    #4. Deltas for the batch
    def _compute_deltas(self, Y_target):
        # Y_target shape, it has the n_output and n_samples
        L = self.L
        delta = [None]*self.L
        out = self.xi[L-1]  # shape (n_out, n_samples)
        hL = self.h[L-1]
        delta[L-1] = self._g_prime(hL, out=out) * (out - Y_target)  # eq. (11)

        # backpropagate
        for l in range(L-1, 1, -1):
            # sum over the next layer deltas weighted by w[l]
            sum_term = self.w[l].T.dot(delta[l])   # shape n[l-1], n_samples
            delta[l-1] = self._g_prime(self.h[l-1], out=self.xi[l-1]) * sum_term
        self.delta = delta
        return delta
    
    # 5. Compute gradients and update weights with momentum
    def _update_weights_from_batch(self, batch_X, batch_y):
        # batch_X shape with n_samples_batch and n_features
        X_mat = batch_X.T
        Y = batch_y.reshape(1, -1) if batch_y.ndim == 1 else batch_y.T
        self._forward(X_mat)
        delta = self._compute_deltas(Y)
        batch_size = batch_X.shape[0]

        for l in range(1, self.L):
            xi_prev = self.xi[l-1]            # (n[l-1], batch_size)
            Dl = delta[l]                     # (n[l], batch_size)
            grad_w = Dl.dot(xi_prev.T)        # (n[l], n[l-1])
            grad_theta = np.sum(Dl, axis=1, keepdims=True)  # (n[l],1)

            dw = - self.lr * grad_w + self.momentum * (self.d_w_prev[l] if self.d_w_prev[l] is not None else 0)
            dtheta = self.lr * grad_theta + self.momentum * (self.d_theta_prev[l] if self.d_theta_prev[l] is not None else 0)

            # apply update for the batch_size
            self.w[l] += dw / batch_size
            self.theta[l] += dtheta / batch_size

            self.d_w_prev[l] = dw
            self.d_theta_prev[l] = dtheta

    # 6. Fit the method, train the loop and split training and validation
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples = X.shape[0]

        # split into train and validation
        if self.val_pct > 0:
            idx = np.arange(n_samples)
            np.random.shuffle(idx)
            cut = int((1.0 - self.val_pct) * n_samples)
            train_idx, val_idx = idx[:cut], idx[cut:]
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        n_train = X_train.shape[0]

        for epoch in range(self.epochs):
            perm = np.random.permutation(n_train)
            for start in range(0, n_train, self.batch_size):
                batch_idx = perm[start:start+self.batch_size]
                self._update_weights_from_batch(X_train[batch_idx], y_train[batch_idx])

            # get the losses
            y_hat_tr = self.predict(X_train)
            train_mse = np.mean((y_hat_tr - y_train)**2)
            self.train_loss_hist.append(train_mse)

            if X_val is not None:
                y_hat_val = self.predict(X_val)
                val_mse = np.mean((y_hat_val - y_val)**2)
                self.val_loss_hist.append(val_mse)
            else:
                self.val_loss_hist.append(np.nan)

        return self
    
    # 7. Predict and loss_epochs
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        A = X.T
        for l in range(1, self.L):
            H = self.w[l].dot(A) - self.theta[l]
            A = self._g(H)
        Y = A.T
        if Y.shape[1] == 1:
            return Y.ravel()
        return Y

    def loss_epochs(self):
        arr = np.zeros((len(self.train_loss_hist), 2))
        arr[:,0] = np.array(self.train_loss_hist)
        arr[:,1] = np.array(self.val_loss_hist, dtype=float)
        return arr