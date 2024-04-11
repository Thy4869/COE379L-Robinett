# Model Deployment and Inference

## Deploying Your Model

1. **Load Your Model**: Load your saved trained model using TensorFlow or any other deep learning framework.

2. **Create an API Endpoint**: Use Flask to create an API endpoint where clients can send requests for inference.

3. **Define Inference Logic**: Implement the logic to perform inference using the loaded model. This typically involves preprocessing the input data, passing it through the model, and returning the predictions.

4. **Expose the Endpoint**: Expose the API endpoint to the outside world, either by deploying it on a web server or using a cloud service like AWS Lambda or Google Cloud Functions.

## Using the Model for Inference

1. **Build Docker Image**: Makesure in the directory contains the Docker file, Flask API, and the model directory by using command ```ls -l```. And then build the docker image by using command ```docker build -t <image name>```.
                           If the command is run on the windows terminal, enter ```sudo chown -R $USER:$USER /home/ubuntu/nb-data``` command to change the ownership of the files in your directory to your user account before you build the docker image.

2. **Run Docker Image**: After the Docker image built succesfully, run the Docker image with ```docker run -it --rm -p 5000:5000 <image name>``` to get the Docker image running.
   
3. **Send HTTP Requests**: Open a new windows of terminal to use tools like `curl` or libraries like `requests` in Python to send HTTP POST requests to the API endpoint deployed.

4. **Prepare Input Data**: Prepare your input data according to the model's input requirements. This may involve resizing images file name, normalizing pixel values, or encoding text.

5. **Send Request**: Send a POST request to the API endpoint with the input data in the request body.

6. **Receive Predictions**: Receive the predictions returned by the API endpoint in the response body. Parse the response to extract the predictions.

## Example Requests

### Using Curl

```bash
 curl -X POST -F "image=@/home/ubuntu/nb-data/House_Data/damage/-97.00144_28.622428999999997.jpeg" localhost:5000/predict
```
**Results:**
"predicted_class": "no_damage"
