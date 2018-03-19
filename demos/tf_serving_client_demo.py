from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from grpc.beta import implementations
import tensorflow as tf
from scipy import misc
import numpy as np

# Create stub
host, port = '127.0.0.1', '9000'
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

# Create prediction request object
request = predict_pb2.PredictRequest()

# Specify model name (must be the same as when the TensorFlow serving serving was started)
request.model_spec.name = 'facefeature'

# Initalize prediction
# Specify signature name (should be the same as specified when exporting model)
# request.model_spec.signature_name = "serving_default"

input_size = 160
picture = misc.imread('../resources/full_body_zh1.jpg')[:input_size, :input_size, :]
picture = picture.reshape([1, input_size, input_size, 3])
picture = picture.astype(np.float32)

request.inputs['images'].CopyFrom(
    tf.contrib.util.make_tensor_proto(picture, shape=[1, input_size, input_size, 3]))
request.inputs['is_training'].CopyFrom(tf.contrib.util.make_tensor_proto(False))
# Call the prediction server
result = stub.Predict(request, 10.0)  # 10 secs timeout
print(np.array(result.outputs['scores'].float_val).shape)
