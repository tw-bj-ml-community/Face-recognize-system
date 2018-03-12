from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from grpc.beta import implementations
import tensorflow as tf
import numpy as np

INPUT_SIZE = 160


class ModelService:
    def __init__(self, host='127.0.0.1', port='9000', model_name='facefeature'):
        # host, port = '127.0.0.1', '9000'
        self.channel = implementations.insecure_channel(host, int(port))
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)
        self.model_name = model_name

    def predict(self, picture):
        #prewhiten, this is used in training, so we need this on prediction.
        picture = self.prewhiten(picture)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.inputs['images'].CopyFrom(
            tf.contrib.util.make_tensor_proto(picture, shape=[1, INPUT_SIZE, INPUT_SIZE, 3]))
        request.inputs['is_training'].CopyFrom(tf.contrib.util.make_tensor_proto(False))  # always false for predict
        # Call the prediction server
        result = self.stub.Predict(request, 10.0)  # 10 secs timeout
        return np.array(result.outputs['scores'].float_val)

    def prewhiten(self, picture):
        mean = np.mean(picture)
        std = np.std(picture)
        std_adj = np.maximum(std, 1.0 / np.sqrt(picture.size))
        picture = np.multiply(np.subtract(picture, mean), 1 / std_adj)
        return picture
