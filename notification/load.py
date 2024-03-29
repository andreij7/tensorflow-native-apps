import tensorflow as tf
import numpy as np

# Load TFLite model and allocate tensors.
interpreter = tf.contrib.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Test model on random input data.
input_shape = input_details[0]['shape']

# change the following line to feed into your own data.
#input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

input_data = np.array([[1,1,1,0,25,25,25,50,25,25,25,75]], dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

tflite_results = interpreter.get_tensor(output_details[1]['index'])
print(tflite_results)
