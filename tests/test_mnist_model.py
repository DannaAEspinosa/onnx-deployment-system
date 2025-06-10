import unittest
import onnxruntime as ort
import numpy as np
from tensorflow.keras.datasets import mnist
from PIL import Image

def preprocess(img_array):
    # Convertir a escala de grises (aunque ya lo es), redimensionar y normalizar
    img = Image.fromarray(img_array).convert('L').resize((28, 28))
    img_arr = np.array(img).astype(np.float32) / 255.0
    img_arr = img_arr.reshape(1, 1, 28, 28)
    return img_arr

class TestMNISTModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Cargar modelo ONNX (debe coincidir con el nombre usado en CI/CD)
        cls.model_path = "mnist12.onnx"
        cls.session = ort.InferenceSession(cls.model_path, providers=['CPUExecutionProvider'])
        cls.input_name = cls.session.get_inputs()[0].name

        # Cargar dataset de prueba
        (_, _), (cls.x_test, cls.y_test) = mnist.load_data()

    def test_prediction_output_type(self):
        # Verifica que la salida sea un int
        input_tensor = preprocess(self.x_test[0])
        output = self.session.run(None, {self.input_name: input_tensor})
        predicted_class = int(np.argmax(output[0]))
        self.assertIsInstance(predicted_class, int)

    def test_prediction_accuracy_threshold(self):
        # Verifica que el modelo tenga al menos 60% de precisión en 10 ejemplos
        correct = 0
        for i in range(10):
            input_tensor = preprocess(self.x_test[i])
            output = self.session.run(None, {self.input_name: input_tensor})
            predicted_class = int(np.argmax(output[0]))
            if predicted_class == int(self.y_test[i]):
                correct += 1
        print(f"Correct predictions: {correct}/10")
        self.assertGreaterEqual(correct, 6)

if __name__ == '__main__':
    unittest.main()
