import tensorflow as tf

def convert_model_to_tflite(keras_model_path: str, tflite_model_path: str) -> None:
    """Convert a Keras model to TensorFlow Lite format."""
    try:
        # Load the Keras model
        model = tf.keras.models.load_model(keras_model_path)
        
        # Create a TFLiteConverter object from the Keras model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Convert the model to TFLite format
        tflite_model = converter.convert()
        
        # Save the TFLite model to the specified path
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model saved to {tflite_model_path}")
    except Exception as e:
        print(f'Error converting model: {str(e)}')
        return False

if __name__ == "__main__":
    keras_model_path = 'model/MultiBranchFusionNet-0.keras' 
    tflite_model_path = 'model/MultiBranchFusionNet-0.tflite'
    convert_model_to_tflite(keras_model_path, tflite_model_path)