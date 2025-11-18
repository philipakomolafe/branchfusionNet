import tensorflow as tf
import os

def convert_model_to_tflite(keras_model_path: str, tflite_model_path: str) -> bool:
    """Convert a Keras model to TensorFlow Lite format with proper bfloat16 handling."""
    try:
        print(f"Loading model from: {keras_model_path}")
        
        # First, try to load with original bfloat16 policy, then convert
        print("Setting mixed_bfloat16 policy to load model correctly...")
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
        
        # Load the Keras model with its original precision policy
        model = tf.keras.models.load_model(keras_model_path)
        print("✅ Model loaded with bfloat16 policy")
        
        # Now switch to float32 for conversion
        print("Switching to float32 policy for TFLite conversion...")
        tf.keras.mixed_precision.set_global_policy('float32')
        
        # Clone model with float32 precision
        model_config = model.get_config()
        model_weights = model.get_weights()
        
        # Recreate model with float32 policy
        model_float32 = tf.keras.Model.from_config(model_config)
        model_float32.set_weights(model_weights)
        
        print("✅ Model converted to float32 precision")
        
        # Create a TFLiteConverter object from the float32 model
        converter = tf.lite.TFLiteConverter.from_keras_model(model_float32)
        
        # Handle unsupported operations by enabling TensorFlow Select operations
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # Default TFLite ops
            tf.lite.OpsSet.SELECT_TF_OPS     # TensorFlow ops fallback
        ]
        
        # Force float32 data types
        converter.target_spec.supported_types = [tf.float32]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        
        # Disable experimental features that may cause issues with bfloat16 models
        converter.experimental_new_converter = False
        converter.experimental_new_quantizer = False
        
        print("Converting model to TFLite format...")
        # Convert the model to TFLite format
        tflite_model = converter.convert()
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(tflite_model_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save the TFLite model to the specified path
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ TFLite model saved to {tflite_model_path}")
        
        # Verify the converted model
        try:
            # Suppress deprecation warnings for verification
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
                interpreter.allocate_tensors()
                
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                model_size_mb = os.path.getsize(tflite_model_path) / 1024 / 1024
                
                print(f"📊 Model size: {model_size_mb:.1f} MB")
                print(f"🔢 Input shape: {input_details[0]['shape']}")
                print(f"🔢 Input dtype: {input_details[0]['dtype']}")
                print(f"🔢 Output shape: {output_details[0]['shape']}")
                print(f"🔢 Output dtype: {output_details[0]['dtype']}")
            
        except Exception as verify_error:
            print(f"⚠️ Model verification failed: {verify_error}")
        
        return True
        
    except Exception as e:
        print(f'❌ Error converting model: {str(e)}')
        
        # Try alternative conversion method
        try:
            print("🔄 Trying alternative conversion method...")
            
            # Reset policy and try simpler approach
            tf.keras.mixed_precision.set_global_policy('float32')
            
            # Load model again with float32 policy
            with tf.keras.utils.custom_object_scope({}):
                model = tf.keras.models.load_model(keras_model_path)
            
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # Minimal converter settings for problematic models
            converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
            converter.target_spec.supported_types = [tf.float32]
            
            # Disable all optimizations for fallback
            tflite_model = converter.convert()
            
            with open(tflite_model_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"✅ Alternative conversion successful: {tflite_model_path}")
            return True
            
        except Exception as fallback_error:
            print(f'❌ Alternative conversion also failed: {str(fallback_error)}')
            
            # Final attempt: Save as SavedModel first, then convert
            try:
                print("🔄 Final attempt: Converting via SavedModel...")
                
                # Save as SavedModel format first
                saved_model_path = tflite_model_path.replace('.tflite', '_savedmodel')
                
                tf.keras.mixed_precision.set_global_policy('float32')
                model = tf.keras.models.load_model(keras_model_path)
                model.save(saved_model_path, save_format='tf')
                
                # Convert from SavedModel
                converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
                converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
                
                tflite_model = converter.convert()
                
                with open(tflite_model_path, 'wb') as f:
                    f.write(tflite_model)
                
                # Clean up temporary SavedModel
                import shutil
                if os.path.exists(saved_model_path):
                    shutil.rmtree(saved_model_path)
                
                print(f"✅ SavedModel conversion successful: {tflite_model_path}")
                return True
                
            except Exception as final_error:
                print(f'❌ All conversion methods failed: {str(final_error)}')
                return False

if __name__ == "__main__":
    keras_model_path = 'model/MultiBranchFusionNet-0.keras' 
    tflite_model_path = 'model/MultiBranchFusionNet-0.tflite'
    
    success = convert_model_to_tflite(keras_model_path, tflite_model_path)
    if not success:
        print("❌ Model conversion failed")
    else:
        print("🎉 Model conversion completed successfully")