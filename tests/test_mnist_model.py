import unittest
import onnxruntime as ort
import numpy as np
from tensorflow.keras.datasets import mnist
import io

def preprocess(img_array):
    img = Image.fromarray(img_array).convert('L').resize((28, 28))
    img_arr = np.array(img).astype(np.float32)
    img_arr /= 255.0
    img_arr = img_arr.reshape(1, 1, 28, 28)
    return img_arr

class TestMNISTModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Cargar modelo ONNX
        cls.session = ort.InferenceSession("model.onnx")
        cls.input_name = cls.session.get_inputs()[0].name

        # Cargar dataset de prueba (solo 1 imagen)
        (_, _), (cls.x_test, cls.y_test) = mnist.load_data()
        cls.x_test = cls.x_test.astype(np.float32)
        cls.x_test = (cls.x_test) / 255.0  # Inversión de color + normalización
        cls.x_test = cls.x_test.reshape(-1, 1, 28, 28)

    def test_prediction_output(self):
        input_tensor = self.x_test[0:1]
        pred = self.session.run(None, {self.input_name: input_tensor})
        pred_class = int(np.argmax(pred[0]))

        self.assertIsInstance(pred_class, int)

    def test_prediction_accuracy_threshold(self):
        # Probar que al menos 60% de las primeras 10 predicciones son correctas
        correct = 0
        for i in range(10):
            input_tensor = self.x_test[i:i+1]
            pred = self.session.run(None, {self.input_name: input_tensor})
            pred_class = int(np.argmax(pred[0]))
            if pred_class == int(self.y_test[i]):
                correct += 1

        self.assertGreaterEqual(correct, 6)  # Al menos 60% (6/10)

if __name__ == '__main__':
    unittest.main()
