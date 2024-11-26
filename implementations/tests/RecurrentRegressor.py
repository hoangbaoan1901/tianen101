import sklearn
import tensorflow as tf



class RecurrentRegressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
	def __init__(self,
				 lstm: list = None,
				 gru: list = None,
				 bilstm: list = None,
				 bigru: list = None,
				 dense: list = None,
				 dropout: list = None,
				 kernel_regularizer: str = None,
				 patience: int = 0,
				 batch_size: int = 32,
				 epochs: int = 1000
				 ):
		"""
		Params:
			All the params are lists, indicating the number of units in each layer. All the list must have the same length - which is the number of the layers. Each index must only have one available value.
			Supporting lstm, gru, bilstm, bigru, dense, dropout.
		"""
		try:
			assert len(lstm) == len(gru) == len(bilstm) == len(
				bigru) == len(dense) == len(dropout)
		except AssertionError:
			raise ValueError("All the params must have the same length")
		self.model = tf.keras.models.Sequential()
		no_layers = len(lstm)
		# Find the last recurrent layer:
		last_recurrent_layer = None
		for i in range(no_layers - 1, -1, -1):
			if lstm[i] is not None or gru[i] is not None or bilstm[i] is not None or bigru[i] is not None:
				last_recurrent_layer = i
				break
		layer_list = []
		for i in range(no_layers):
			if i == last_recurrent_layer:
				if lstm[i] is not None:
					layer_list.append(tf.keras.layers.LSTM(
						lstm[i], return_sequences=False, kernel_regularizer=kernel_regularizer))
				elif gru[i] is not None:
					layer_list.append(tf.keras.layers.GRU(
						gru[i], return_sequences=False, kernel_regularizer=kernel_regularizer))
				elif bilstm[i] is not None:
					layer_list.append(tf.keras.layers.Bidirectional(
						tf.keras.layers.LSTM(bilstm[i], return_sequences=False, kernel_regularizer=kernel_regularizer)))
				elif bigru[i] is not None:
					layer_list.append(tf.keras.layers.Bidirectional(
						tf.keras.layers.GRU(bigru[i], return_sequences=False, kernel_regularizer=kernel_regularizer)))
				elif dense[i] is not None:
					layer_list.append(tf.keras.layers.Dense(
						dense[i], kernel_regularizer=kernel_regularizer))
			else:
				if lstm[i] is not None:
					layer_list.append(tf.keras.layers.LSTM(
						lstm[i], return_sequences=True, kernel_regularizer=kernel_regularizer))
				elif gru[i] is not None:
					layer_list.append(tf.keras.layers.GRU(
						gru[i], return_sequences=True, kernel_regularizer=kernel_regularizer))
				elif bilstm[i] is not None:
					layer_list.append(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
						bilstm[i], return_sequences=True, kernel_regularizer=kernel_regularizer)))
				elif bigru[i] is not None:
					layer_list.append(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
						bigru[i], return_sequences=True, kernel_regularizer=kernel_regularizer)))
				elif dense[i] is not None:
					layer_list.append(tf.keras.layers.Dense(
						dense[i], kernel_regularizer=kernel_regularizer))
				if dropout[i] is not None:
					layer_list.append(tf.keras.layers.Dropout(dropout[i]))

		self.architecture = layer_list
		self.model = tf.keras.models.Sequential()
		self.patience = patience
		self.batch_size = batch_size
		self.epochs = epochs

	def fit(self, X, y):
		self.model.add(tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])))
		for layer in self.architecture:
			self.model.add(layer)
		self.model.add(tf.keras.layers.Dense(1))
		self.model.compile(optimizer='adam', loss='mse')
		if self.patience > 0:
			early_stopping = tf.keras.callbacks.EarlyStopping(
				monitor='loss', patience=self.patience, restore_best_weights=True)
			self.model.fit(X, y, batch_size=self.batch_size,
						   epochs=self.epochs, callbacks=[early_stopping])
		else:
			self.model.fit(X, y, batch_size=self.batch_size,
						   epochs=self.epochs)

	def predict(self, X):
		return self.model.predict(X)
	