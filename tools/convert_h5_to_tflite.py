import tensorflow as tf

def convert_h5_to_tflite(file_path: str):
    keras_model = tf.keras.models.load_model(file_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    with open(f"{file_path.replace(".h5", ".tflite")}.tflite", "wb") as f:
        f.write(tflite_model)
    