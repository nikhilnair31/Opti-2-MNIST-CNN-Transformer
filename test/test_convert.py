import tensorflow as tf

# Load the SavedModel
saved_model_path = 'Models/transformer_model/'
model = tf.saved_model.load(saved_model_path)

# Convert to Keras model
keras_model = tf.keras.models.load_model(saved_model_path)

# Save as .h5 file
h5_model_path = 'Models/transformer_model.h5'
keras_model.save(h5_model_path)