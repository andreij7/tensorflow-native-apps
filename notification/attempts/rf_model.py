import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv("notifications.csv")

print(data.head())

y = data.opened
X = data.drop('opened', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.99) #i know the split is high. i produced to much fake data
print("\nX_train:\n")
print(X_train.head())
print(X_train.shape)

print("\nX_test:\n")
print(X_test.head())
print(X_test.shape)

import tensorflow as tf
from tensorflow import keras
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources

tf.reset_default_graph()

# Parameters

num_steps = 1 # Total steps to train
num_classes = 2 
num_features = 12
num_trees = 10
max_nodes = 1000 

# Input and Target placeholders 

X = tf.placeholder(tf.float32, shape=(1,12), name="input")

Y = tf.placeholder(tf.int64, shape=[None], name="output")

var = tf.get_variable("weights", dtype=tf.float32, shape=(1,12))

val = X + var

out = tf.identity(val, name="out")

# Random Forest Parameters

hparams = tensor_forest.ForestHParams(reuse=True, num_classes=num_classes, num_features=num_features, num_trees=num_trees, max_nodes=max_nodes).fill()

# Build the Random Forest

forest_graph = tensor_forest.RandomForestGraphs(hparams)

# Get training graph and loss

train_op = forest_graph.training_graph(X, Y)

loss_op = forest_graph.training_loss(X, Y)

# Measure the accuracy

infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the variables (i.e. assign their default value) and forest resources

init_vars = tf.group(tf.global_variables_initializer(), resources.initialize_resources(resources.shared_resources()))
saver = tf.train.Saver()
# Start TensorFlow session

with tf.Session() as sess:
    # Run the initializer
    sess.run(init_vars)

    percentDone = 0
    # Training
    for i in range(1, num_steps + 1):
        total = 0
        correctTotal = 0

        for index in range(X_train.shape[0]):
            X_train_row = X_train.iloc[index].values
            y_train_row = y_train.iloc[index]

            X_train_reshaped = np.reshape(X_train_row, (1, 12))

            _, l = sess.run([train_op, loss_op], feed_dict={X: X_train_reshaped, Y: [y_train_row]})

            if index % 2000 == 0:
                acc = sess.run(accuracy_op, feed_dict={X: X_train_reshaped, Y: [y_train_row]})
                total = total + 1
                if(int(acc) > 0):
                    correctTotal = correctTotal + 1

                overallAcc = 0
                if(correctTotal > 0):
                    overallAcc = (correctTotal/total) * 100

                if index % 20000 == 0:
                    print('Step %i, Loss: %f, Acc: %f Total Accuracy: %f' % (i, l, acc, overallAcc))

    converter = tf.lite.TFLiteConverter.from_session(sess, [X], [out, Y])
    tflite_model = converter.convert()
    open("notifications.tflite", "wb").write(tflite_model)
    model_path = 'savermodel'
    save_path = saver.save(sess, model_path)

