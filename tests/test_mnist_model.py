from PIL import Image
import numpy as np
import onnxruntime as ort
import unittest
from tensorflow.keras.datasets import mnist
import io

def preprocess(img_array):
    img = Image.fromarray(img_array).convert('L').resize((28, 28))
    img_arr = np.array(img).astype(np.float32)
    img_arr = 255 - img_arr  # Invertir colores si el fondo es blanco
    img_arr /= 255.0
    img_arr = img_arr.reshape(1, 1, 28, 28)
    return img_arr

class TestMNISTModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = ort.InferenceSession("model.onnx")
        (cls.x_test, cls.y_test), _ = mnist.load_data()

    def test_model_outputs(self):
        # Simple prueba con una imagen
        input_tensor = preprocess(self.x_test[0])
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: input_tensor})
        predicted = np.argmax(output[0])
        self.assertIsInstance(predicted, int)

    def test_prediction_accuracy_threshold(self):
        input_name = self.session.get_inputs()[0].name
        correct = 0
        for i in range(10):
            img = preprocess(self.x_test[i])
            pred = np.argmax(self.session.run(None, {input_name: img})[0])
            if pred == self.y_test[i]:
                correct += 1
        print(f"Correct predictions: {correct}/10")
        self.assertGreaterEqual(correct, 6)  # 60%
