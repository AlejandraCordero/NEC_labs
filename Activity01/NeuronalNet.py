import numpy as np

class NeuralNet:
    def __init__(self, n_units, lr=0.01, momentum=0.9, epochs=200,
                 activation="tanh", val_pct=0.2, batch_size=32, random_state=42):
        """
        Initialize Neural Network with Back-Propagation

        Parameters:
        -----------
        n_units : list
            List with number of units in each layer (including input and output)
            Example: [10, 64, 32, 1] means 10 inputs, 2 hidden layers (64, 32), 1 output
        lr : float
            Learning rate (η)
        momentum : float
            Momentum term (α)
        epochs : int
            Number of training epochs
        activation : str
            Activation function: 'sigmoid', 'relu', 'tanh', 'linear'
        val_pct : float
            Percentage of data to use for validation (0.0 to 1.0)
        batch_size : int
            Number of patterns per batch
        random_state : int
            Random seed for reproducibility
        """
        np.random.seed(random_state)

        self.n_units = n_units
        self.L = len(n_units)  # Number of layers
        self.lr = lr
        self.momentum = momentum
        self.epochs = epochs
        self.activation_name = activation
        self.val_pct = val_pct
        self.batch_size = batch_size
        self.loss_history = []

    # ------------------------------------------------------
    # ACTIVATIONS
    # ------------------------------------------------------
    def sigmoid(self, h):
        """Sigmoid activation: g(h) = 1 / (1 + e^(-h))"""
        h = np.clip(h, -500, 500)
        return 1.0 / (1.0 + np.exp(-h))

    def sigmoid_deriv(self, h):
        """Derivative of sigmoid: g'(h) = g(h) * (1 - g(h))"""
        g = self.sigmoid(h)
        return g * (1.0 - g)

    def tanh(self, h):
        """Hyperbolic tangent activation"""
        return np.tanh(h)

    def tanh_deriv(self, h):
        """Derivative of tanh: g'(h) = 1 - tanh(h)^2"""
        return 1.0 - np.tanh(h)**2

    def relu(self, h):
        """ReLU activation: g(h) = max(0, h)"""
        return np.maximum(0, h)

    def relu_deriv(self, h):
        """Derivative of ReLU"""
        return (h > 0).astype(float)

    def linear(self, h):
        """Linear activation: g(h) = h"""
        return h

    def linear_deriv(self, h):
        """Derivative of linear activation"""
        return np.ones_like(h)

    def activate(self, h):
        """Apply activation function"""
        if self.activation_name == "relu":
            return self.relu(h)
        elif self.activation_name == "tanh":
            return self.tanh(h)
        elif self.activation_name == "linear":
            return self.linear(h)
        else:  # sigmoid
            return self.sigmoid(h)

    def activate_deriv(self, h):
        """Apply derivative of activation function"""
        if self.activation_name == "relu":
            return self.relu_deriv(h)
        elif self.activation_name == "tanh":
            return self.tanh_deriv(h)
        elif self.activation_name == "linear":
            return self.linear_deriv(h)
        else:  # sigmoid
            return self.sigmoid_deriv(h)

    # ------------------------------------------------------
    # INIT WEIGHTS AND THRESHOLDS
    # ------------------------------------------------------
    def init_weights(self):
        """
        Initialize weights (w) and thresholds (theta) using Xavier initialization
        Following BP.v2 specification with proper variable names
        """
        self.w = []      # w[ℓ][i,j] = weight from unit j in layer ℓ-1 to unit i in layer ℓ
        self.theta = []  # theta[ℓ][i] = threshold of unit i in layer ℓ

        # Initialize previous weight changes for momentum
        self.d_w_prev = []
        self.d_theta_prev = []

        # w[0] is not used (no weights before first layer)
        self.w.append(None)
        self.theta.append(None)
        self.d_w_prev.append(None)
        self.d_theta_prev.append(None)

        # Initialize weights for layers 2 to L
        for ell in range(1, self.L):
            n_in = self.n_units[ell - 1]   # Units in previous layer
            n_out = self.n_units[ell]      # Units in current layer

            # Xavier/He initialization
            if self.activation_name == "relu":
                scale = np.sqrt(2.0 / n_in)
            else:
                scale = np.sqrt(2.0 / (n_in + n_out))

            # w[ℓ] is a matrix of size (n_out, n_in)
            self.w.append(np.random.randn(n_out, n_in) * scale)
            # theta[ℓ] is an array of size n_out
            self.theta.append(np.random.randn(n_out) * 0.01)

            # Initialize momentum terms to zero
            self.d_w_prev.append(np.zeros((n_out, n_in)))
            self.d_theta_prev.append(np.zeros(n_out))

    # ------------------------------------------------------
    # FEED-FORWARD PROPAGATION
    # ------------------------------------------------------
    def forward(self, x):
        """
        Feed-forward propagation following BP.v2 equations

        Parameters:
        -----------
        x : numpy array
            Input pattern (can be 1D for single pattern or 2D for batch)

        Returns:
        --------
        o : numpy array
            Output prediction
        """
        # Ensure x is 2D (batch_size, n_features)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Store activations for backpropagation
        # xi[ℓ] stores activations ξ^(ℓ) for layer ℓ
        self.xi = []
        # h[ℓ] stores fields h^(ℓ) for layer ℓ
        self.h = []

        # Layer 1 (input): ξ^(1) = x
        self.xi.append(x)
        self.h.append(None)  # No fields for input layer

        # Propagate through layers 2 to L
        for ell in range(1, self.L):
            # h^(ℓ)_i = Σ_j w^(ℓ)_ij * ξ^(ℓ-1)_j - θ^(ℓ)_i
            # Matrix form: h^(ℓ) = ξ^(ℓ-1) @ w^(ℓ).T - θ^(ℓ)
            h_ell = self.xi[ell - 1] @ self.w[ell].T - self.theta[ell]

            # For output layer, use linear activation
            if ell == self.L - 1:
                xi_ell = h_ell  # Linear output
            else:
                # ξ^(ℓ)_i = g(h^(ℓ)_i)
                xi_ell = self.activate(h_ell)

            self.h.append(h_ell)
            self.xi.append(xi_ell)

        # Output: o(x) = ξ^(L)
        return self.xi[-1]

    # ------------------------------------------------------
    # ERROR BACK-PROPAGATION
    # ------------------------------------------------------
    def backward(self, z):
        """
        Back-propagation of error following BP.v2 equations

        Parameters:
        -----------
        z : numpy array
            Desired output (target values)

        This method computes the delta values (Δ) for all layers
        """
        # Ensure z is 2D
        if z.ndim == 1:
            z = z.reshape(-1, 1)

        batch_size = z.shape[0]

        # Store delta values: delta[ℓ] = Δ^(ℓ) for layer ℓ
        self.delta = []

        # delta[0] is not used (no delta for input layer)
        self.delta.append(None)

        # Compute delta for output layer (L)
        # Δ^(L)_i = g'(h^(L)_i) * (o_i(x) - z_i)
        # For linear output: g'(h) = 1, so Δ^(L) = (o - z)
        o = self.xi[-1]  # Output predictions
        delta_L = (o - z)  # Linear output layer
        self.delta.append(delta_L)

        # Back-propagate deltas from layer L-1 down to layer 2
        for ell in range(self.L - 2, 0, -1):
            # Δ^(ℓ)_j = g'(h^(ℓ)_j) * Σ_i Δ^(ℓ+1)_i * w^(ℓ+1)_ij
            # Matrix form: Δ^(ℓ) = g'(h^(ℓ)) ⊙ (Δ^(ℓ+1) @ w^(ℓ+1))
            delta_next = self.delta[ell + 1 - (self.L - len(self.delta))]
            delta_ell = self.activate_deriv(self.h[ell]) * (delta_next @ self.w[ell + 1])
            self.delta.insert(1, delta_ell)  # Insert at beginning (after None)

    def compute_weight_updates(self):
        """
        Compute weight and threshold updates using deltas
        Following BP.v2 equations (14)

        Returns:
        --------
        d_w : list of arrays
            Weight updates δw^(ℓ)
        d_theta : list of arrays
            Threshold updates δθ^(ℓ)
        """
        d_w = []
        d_theta = []

        batch_size = self.xi[0].shape[0]

        # d_w[0] and d_theta[0] are not used
        d_w.append(None)
        d_theta.append(None)

        # Compute updates for layers 2 to L
        for ell in range(1, self.L):
            # δw^(ℓ)_ij = -η * Δ^(ℓ)_i * ξ^(ℓ-1)_j + α * δw^(ℓ)_ij(prev)
            # Matrix form: δw^(ℓ) = -η * Δ^(ℓ).T @ ξ^(ℓ-1) / batch_size + α * δw^(ℓ)(prev)
            delta_ell = self.delta[ell]
            xi_prev = self.xi[ell - 1]

            grad_w = (delta_ell.T @ xi_prev) / batch_size
            dw = -self.lr * grad_w + self.momentum * self.d_w_prev[ell]

            # δθ^(ℓ)_i = η * Δ^(ℓ)_i + α * δθ^(ℓ)_i(prev)
            grad_theta = np.mean(delta_ell, axis=0)
            dtheta = self.lr * grad_theta + self.momentum * self.d_theta_prev[ell]

            # Clip gradients to prevent exploding gradients
            dw = np.clip(dw, -10, 10)
            dtheta = np.clip(dtheta, -10, 10)

            d_w.append(dw)
            d_theta.append(dtheta)

        return d_w, d_theta

    # ------------------------------------------------------
    # FIT - TRAINING WITH ONLINE/BATCH BP
    # ------------------------------------------------------
    def fit(self, X, y):
        """
        Train the neural network using Back-Propagation
        Following BP.v2 online/batch algorithm

        Parameters:
        -----------
        X : numpy array (n_samples, n_features)
            Training data
        y : numpy array (n_samples,) or (n_samples, 1)
            Target values
        """
        # Ensure inputs are numpy arrays
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Split train/validation
        N = len(X)
        idx = np.random.permutation(N)
        split = int(N * (1 - self.val_pct))

        train_idx = idx[:split]
        val_idx = idx[split:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Initialize weights and thresholds
        # First element of n_units should be number of input features
        if self.n_units[0] != X.shape[1]:
            self.n_units[0] = X.shape[1]
            self.L = len(self.n_units)

        self.init_weights()
        self.loss_history = []

        # Training loop (epochs)
        for epoch in range(self.epochs):
            # Shuffle training data for each epoch
            perm = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]

            # Mini-batch training
            n_batches = int(np.ceil(len(X_train) / self.batch_size))

            for batch in range(n_batches):
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, len(X_train))

                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]

                # Forward propagation
                o_batch = self.forward(X_batch)

                # Backward propagation
                self.backward(y_batch)

                # Compute weight updates
                d_w, d_theta = self.compute_weight_updates()

                # Update weights and thresholds
                # w^(ℓ) ← w^(ℓ) + δw^(ℓ)
                # θ^(ℓ) ← θ^(ℓ) + δθ^(ℓ)
                for ell in range(1, self.L):
                    self.w[ell] += d_w[ell]
                    self.theta[ell] += d_theta[ell]

                    # Store for momentum in next iteration
                    self.d_w_prev[ell] = d_w[ell]
                    self.d_theta_prev[ell] = d_theta[ell]

            # Compute training and validation losses for monitoring
            y_train_pred = self.forward(X_train)
            train_loss = np.mean((y_train_pred - y_train) ** 2)

            if len(X_val) > 0:
                y_val_pred = self.forward(X_val)
                val_loss = np.mean((y_val_pred - y_val) ** 2)
            else:
                val_loss = train_loss

            self.loss_history.append([train_loss, val_loss])

            # Optional: Print progress every 20 epochs
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")

    # ------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------
    def predict(self, X):
        """
        Make predictions for input data X

        Parameters:
        -----------
        X : numpy array (n_samples, n_features)
            Input data

        Returns:
        --------
        predictions : numpy array (n_samples,)
            Predicted values
        """
        X = np.array(X, dtype=np.float64)
        y_pred = self.forward(X)
        return y_pred.flatten()

    def loss_epochs(self):
        """
        Get the training and validation loss history

        Returns:
        --------
        loss_history : numpy array (n_epochs, 2)
            Array with [train_loss, val_loss] for each epoch
        """
        return np.array(self.loss_history)
