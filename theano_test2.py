import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(64, input_dim=20, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(10, init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
x_train = np.random.random((1000, 20))
y_train = np.random.random((1000, 10))
model.fit(x_train, y_train,nb_epoch=20,batch_size=16)
x_test = np.random.random((0, 20))
y_test = np.random.random((0, 10))
score = model.evaluate(x_test, y_test, batch_size=16)

