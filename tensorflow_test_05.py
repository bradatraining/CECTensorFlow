'''

This a modification of the tensorflow quickstart, showing
some of the intermediate steps, so you can see the images, and
some of the probabilities.

https://www.tensorflow.org/tutorials/quickstart/beginner

License on the code from above is Apache 2.0
https://www.apache.org/licenses/LICENSE-2.0

2020-May-15
Andrew Anselmo
andrew@clipboardengineering.com

Added a simple image viewer
'''

import tensorflow as tf
import numpy as np
from numpy import newaxis


chars_to_use=[' ','.',',',':','-','+','x','%','@','#']
scaling=(len(chars_to_use)-1)/255

def print_nicely(array_input):
    for i in range(0,28):
        for j in range(0,28):
            val=array_input[0,i,j]
            print(chars_to_use[int(val*scaling)],end="")
        print()

print("---------------------")
print("Getting dataset.")

# Get the dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_raw = x_train
x_test_raw = x_test

x_train, x_test = x_train / 255.0, x_test / 255.0

# Show what this image looks like, really, really simply

# index is the data element, 0 to 59999
# make a simple map of non-zero elements

index=5
print("index from training data set, 0 to 59999 = " + str(index))
print("true value = " + str(y_train[index]))
print("x_train image is:")
train_val_raw=x_train_raw[index][newaxis,:,:]
print(" ")
print_nicely(train_val_raw)


# Create the model
model = tf.keras.models.Sequential([
tf.keras.layers.Flatten(input_shape=(28, 28)),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(10)
])


# Check to see, based on a random test value, the
# probabilities
test_index=566
print("test_index = " + str(test_index))
test_val_raw=x_test_raw[test_index][newaxis,:,:]
test_val=x_test[test_index][newaxis,:,:]
actual_val=y_test[test_index]

print("showing x_test image:")
print(" ")
print_nicely(test_val_raw)


print("real val from y_test = " + str(actual_val))

print("---------------------")
print("Before modeling...")

predictions = model(test_val).numpy()

print("predictions")
print(predictions)

print("probabilities")
probabilities = tf.nn.softmax(predictions).numpy()
for i in range(0,10):
        print("i, probability of being that digit = " + str(i) + ", " +  str(probabilities[0][i]))

print("---------------------")
print("Performing modeling...")

print("Current loss functions:")
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

for i in range(0,10):
    print("i, loss = " + str(i) + "," +  str(loss_fn(i, predictions).numpy()))

# fit
print("Fitting...")
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1)

model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()])

nerror=0
test_range=range(1000,1050)
for test_index in test_range:
    print(" ")
    print("test_index = " + str(test_index))
    test_val_raw=x_test_raw[test_index][newaxis,:,:]
    test_val=x_test[test_index][newaxis,:,:]
    actual_val=y_test[test_index]

    print("testing value:")

    print("showing x_test image:")

    print(" ")
    print_nicely(test_val_raw)


    print("real val from y_test = " + str(actual_val))
    print("probability of this being that digit:")
    prob=probability_model(test_val)
    array_of_probabilities=prob.numpy()[0]
    for i in range(0,10):
        print("i, prob of being that digit = " + str(i) + ", " +  str(prob.numpy()[0][i]))

    est_digit = np.where(array_of_probabilities==np.amax(array_of_probabilities))[0][0]

    print("Digit, via model is estimated to be: "  + str(est_digit))
    if(est_digit != actual_val):
            print("ERROR in estimation!")
            nerror=nerror+1

print("Total tested:" + str(len(test_range)))
print("Final error count:" + str(nerror))
print("Success probabilities:")
print("From individual testing: " + str((float(len(test_range))-float(nerror))/float(len(test_range))))
model_val=model.evaluate(x_test,  y_test, verbose=2)[1]
print("From model function: " + str(model_val))



