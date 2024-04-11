from flask import Flask, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# LOADING THE MODEL YIPPEEEE
model_ANN = tf.keras.models.load_model('models/imagesANN_1.keras')
model_ANN_2 = tf.keras.models.load_model('models/imagesANN_2.keras')
model_CNN = tf.keras.models.load_model('models/imagesCNN_1.keras')
model_altCNN = tf.keras.models.load_model('models/images_altCNN_1.keras')

# Create a general get route to explain what this flask app does. List the models and their versions, as well as the routes avaliable
@app.route('/', methods=['GET'])
def home():
   return {
      "message": "Welcome to the Image Classification API! Here are the available routes:",
      "routes": [
         "/models/best - Best model (Alternative CNN) to classify images",
         "/models/imagesANN/v1 - Dense ANN model",
         "/models/imagesANN/v2 - Dense ANN model 2",
         "/models/imagesCNN/v1 - CNN model",
         "/models/images_altCNN/v1 - Alternative CNN model",
      ]
   }

# Create a route to talk about the model imagesANN_1.keras
@app.route('/models/imagesANN/v1', methods=['GET'])
def model_info_ANN():
   return {
      "version": "v1",
      "name": "imagesANN",
      "description": "Classify images that have damage or not using a dense ANN (Artificial Neural Network)",
      "number_of_parameters": 6316417
   }

# Create a route to talk about the model imagesANN_2.keras
@app.route('/models/imagesANN/v2', methods=['GET'])
def model_info_ANN_2():
   return {
      "version": "v2",
      "name": "imagesANN",
      "description": "Classify images that have damage or not using a dense ANN (Artificial Neural Network) with a different architecture",
      "number_of_parameters": 12624385
   }

# Create a route to talk about the model imagesCNN_1.keras
@app.route('/models/imagesCNN/v1', methods=['GET'])
def model_info_CNN():
   return {
      "version": "v1",
      "name": "imagesCNN",
      "description": "Classify images that have damage or not using a CNN (Convolutional Neural Network)",
      "number_of_parameters": 1627961
   }

# Create a route to talk about the model images_altCNN_1.keras
@app.route('/models/images_altCNN/v1', methods=['GET'])
def model_info_altCNN():
   return {
      "version": "v1",
      "name": "images_altCNN",
      "description": "Classify images that have damage or not using an alternative CNN (Convolutional Neural Network)",
      "number_of_parameters": 2601153
   }

# Create a route to talk about the model images_altCNN_1.keras
@app.route('/models/best', methods=['GET'])
def model_infobest():
   return {
      "version": "v1",
      "name": "images_altCNN (best)",
      "description": "Classify images that have damage or not using an alternative CNN (Convolutional Neural Network)",
      "number_of_parameters": 2601153
   }

def preprocess_input(im):
    """
    Converts user-provided input into an array that can be used with the model.
    This function could raise an exception.
    """
    # Convert to a numpy array
    d = np.array(im)
    # then add an extra dimension, because Keras expects a list of objects,
    # so padding with an extra dimension
    return d.reshape(1, 128, 128, 3)

# Pass the image to the model and return the prediction
@app.route('/models/imagesANN/v1', methods=['POST'])
def classify_image_ANN():
   im = request.json.get('image')
   if not im:
      return {"error": "The `image` field is required"}, 404
   try:
      data = preprocess_input(im)
   except Exception as e:
      return {"error": f"Could not process the `image` field; details: {e}"}, 404
   # Converted result of model to list because numpy array is returned ans is not serializable in json, so regular list is returned
   return { "result": model_ANN.predict(data).tolist()}

# Pass the image to the model and return the prediction
@app.route('/models/imagesANN/v2', methods=['POST'])
def classify_image_ANN_2():
   im = request.json.get('image')
   if not im:
      return {"error": "The `image` field is required"}, 404
   try:
      data = preprocess_input(im)
   except Exception as e:
      return {"error": f"Could not process the `image` field; details: {e}"}, 404
   # Converted result of model to list because numpy array is returned ans is not serializable in json, so regular list is returned
   return { "result": model_ANN_2.predict(data).tolist()}

# Pass the image to the model and return the prediction
@app.route('/models/imagesCNN/v1', methods=['POST'])
def classify_image_CNN():
   im = request.json.get('image')
   if not im:
      return {"error": "The `image` field is required"}, 404
   try:
      data = preprocess_input(im)
   except Exception as e:
      return {"error": f"Could not process the `image` field; details: {e}"}, 404
   # Converted result of model to list because numpy array is returned ans is not serializable in json, so regular list is returned
   return { "result": model_CNN.predict(data).tolist()}

# Pass the image to the model and return the prediction
@app.route('/models/images_altCNN/v1', methods=['POST'])
def classify_image_altCNN():
   im = request.json.get('image')
   if not im:
      return {"error": "The `image` field is required"}, 404
   try:
      data = preprocess_input(im)
   except Exception as e:
      return {"error": f"Could not process the `image` field; details: {e}"}, 404
   # Converted result of model to list because numpy array is returned ans is not serializable in json, so regular list is returned
   return { "result": model_altCNN.predict(data).tolist()}

# Pass the image to the best model (altCNN) and return the prediction
@app.route('/models/best', methods=['POST'])
def classify_image_best():
   im = request.json.get('image')
   if not im:
      return {"error": "The `image` field is required"}, 404
   try:
      data = preprocess_input(im)
   except Exception as e:
      return {"error": f"Could not process the `image` field; details: {e}"}, 404
   # Converted result of model to list because numpy array is returned ans is not serializable in json, so regular list is returned
   return { "result": model_altCNN.predict(data).tolist()}

# start the development server
if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')