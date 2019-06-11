import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

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

tf.enable_eager_execution()

tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(123)

NUMERIC_COLUMNS = ['isAm','isActiveDayOfWeek','isGeneric','isInInterest','amOPR','pmOPR','activeDayOPR','nonActiveDayOPR','genericDescriptionOPR','uniqueDescriptionOPR','interestOPR','otherTopicOPR']

feature_columns = []

fc = tf.feature_column

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(fc.numeric_column(feature_name, dtype=tf.float32))

NUM_EXAMPLES = len(y_train)
MAX_STEPS = 2

def make_input_fn(X, y, n_epochs=None, shuffle=True):
  def input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
    if shuffle:
      dataset = dataset.shuffle(NUM_EXAMPLES)
    # For training, cycle thru dataset as many times as need (n_epochs=None).    
    dataset = dataset.repeat(n_epochs)  
    # In memory training doesn't use batching.
    dataset = dataset.batch(NUM_EXAMPLES)
    return dataset
  return input_fn

train_input_fn = make_input_fn(X_train, y_train)
eval_input_fn = make_input_fn(X_test, y_test, shuffle=False, n_epochs=1)

'''
print("Build Linear classifier")
linear_est = tf.estimator.LinearClassifier(feature_columns)

print("Train Linear classifier")
# Train model.
linear_est.train(train_input_fn, max_steps=MAX_STEPS)

print("Evaluate")
# Evaluation.
results = linear_est.evaluate(eval_input_fn)
print('Accuracy : ', results['accuracy'])
print('Dummy model: ', results['accuracy_baseline'])
'''
# Since data fits into memory, use entire dataset per layer. It will be faster.
# Above one batch is defined as the entire dataset. 
n_batches = 1
est = tf.estimator.BoostedTreesClassifier(feature_columns, n_batches_per_layer=n_batches)

# The model will stop training once the specified number of trees is built, not 
# based on the number of steps.
print("Train Estimator")
est.train(train_input_fn, max_steps=MAX_STEPS)

print("Evaluate Estimator classifier")
# Eval.
results = est.evaluate(eval_input_fn)
print('Accuracy : ', results['accuracy'])
print('Dummy model: ', results['accuracy_baseline'])

#https://github.com/MtDersvan/tf_playground/blob/master/wide_and_deep_tutorial/wide_and_deep_basic_serving.md

def export_tflite(classifier):
        with tf.Session() as sess:
            # First let's load meta graph and restore weights
            latest_checkpoint_path = classifier.latest_checkpoint()
            saver = tf.train.import_meta_graph(latest_checkpoint_path + '.meta')
            saver.restore(sess, latest_checkpoint_path)

            op = sess.graph.get_operations()
            #[print(m.values()) for m in op][1]

            # Get the input and output tensors
            input_tensor = sess.graph.get_tensor_by_name("dnn/input_from_feature_columns/input_layer/concat:0")
            bias_tensor = sess.graph.get_tensor_by_name("dnn/logits/BiasAdd:0")

            # here the code differs from the toco example above
            sess.run(tf.global_variables_initializer())
            converter = tf.lite.TFLiteConverter.from_session(sess, [input_tensor], [bias_tensor])
            tflite_model = converter.convert()
            open("bt_converted_model.tflite", "wb").write(tflite_model)

export_tflite(est)