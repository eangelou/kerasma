from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("joker_2017.csv", delimiter="\t")
dataset = numpy.append(dataset, numpy.loadtxt("joker_2016.csv", delimiter="\t"), axis=0)
dataset = numpy.append(dataset, numpy.loadtxt("joker_2015.csv", delimiter="\t"), axis=0)
print (dataset)
# split into input (X) and output (Y) variables
X = dataset[:,0:6]
Y = dataset[:,0:6]

# create model
model = Sequential()
model.add(Dense(12, input_dim=6, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(36, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(6, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

