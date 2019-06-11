import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

data = pd.read_csv("raw/notifications.csv")

y = data.opened
X = data.drop('opened', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2) 
print("\nX_train:\n")
print(X_train.shape)

print("\nX_test:\n")
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


est = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[1024, 512, 256])

MAX_STEPS = 200

print("Train DNNClassifier")
est.train(train_input_fn, max_steps=MAX_STEPS)

print("Evaluate DNNClassifier")
# Eval.
results = est.evaluate(eval_input_fn)
print('Accuracy : ', results['accuracy'])
print('Dummy model: ', results['accuracy_baseline'])

#https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier
'''
metrics = estimator.evaluate(input_fn=input_fn_eval)
predictions = estimator.predict(input_fn=input_fn_predict)
'''

NUMERIC_COLUMNS = ['isAm','isActiveDayOfWeek','isGeneric','isInInterest','amOPR','pmOPR','activeDayOPR','nonActiveDayOPR','genericDescriptionOPR','uniqueDescriptionOPR','interestOPR','otherTopicOPR']

testSize = 100
percentageColumn = 100
boolColumn = 2

pInputs = {
    "activeDayOPR": np.random.randint(percentageColumn, size=testSize), 
    "amOPR" : np.random.randint(percentageColumn, size=testSize), 
    "isAm": np.random.randint(boolColumn, size=testSize), 
    "isActiveDayOfWeek": np.random.randint(boolColumn, size=testSize), 
    "isGeneric": np.random.randint(boolColumn, size=testSize), 
    "isInInterest": np.random.randint(percentageColumn, size=testSize), 
    "pmOPR": np.random.randint(percentageColumn, size=testSize), 
    "genericDescriptionOPR": np.random.randint(percentageColumn, size=testSize), 
    "uniqueDescriptionOPR": np.random.randint(percentageColumn, size=testSize), 
    "interestOPR": np.random.randint(percentageColumn, size=testSize), 
    "otherTopicOPR": np.random.randint(percentageColumn, size=testSize), 
    "nonActiveDayOPR": np.random.randint(percentageColumn, size=testSize)
}

predict_input_fn = tf.estimator.inputs.numpy_input_fn(x=pInputs, y=None,shuffle=False)

predictions = list(est.predict(input_fn=predict_input_fn))
predicted_classes = [p["classes"] for p in predictions]

print(
    "New Samples, Class Predictions:    {}\n"
    .format(predicted_classes))

feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)

export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

def export_tflite(classifier):
        with tf.Session() as sess:
            # First let's load meta graph and restore weights
            latest_checkpoint_path = classifier.latest_checkpoint()
            saver = tf.train.import_meta_graph(latest_checkpoint_path + '.meta')
            saver.restore(sess, latest_checkpoint_path)

            #op = sess.graph.get_operations()
            #[print(m.values()) for m in op][1]

            # Get the input and output tensors
            input_tensor = sess.graph.get_tensor_by_name("dnn/input_from_feature_columns/input_layer/concat:0")
            bias_add_tensor = sess.graph.get_tensor_by_name("dnn/logits/BiasAdd:0")
            out_tensor = sess.graph.get_tensor_by_name("dnn/head/predictions/probabilities:0")
            logis_tensor = sess.graph.get_tensor_by_name("dnn/head/predictions/logistic:0")
            two_class = sess.graph.get_tensor_by_name("dnn/head/predictions/two_class_logits:0")

            # here the code differs from the toco example above
            sess.run(tf.global_variables_initializer())
            converter = tf.lite.TFLiteConverter.from_session(sess, [input_tensor], [bias_add_tensor, out_tensor, logis_tensor, two_class])
            tflite_model = converter.convert()
            open("converted_model.tflite", "wb").write(tflite_model)

export_tflite(est)