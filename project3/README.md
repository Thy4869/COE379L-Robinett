# **README**
*James Grant Robinett (jgr2722) - Hung-Yi Tseng (ht7796)*

## **COE 379L - Project 3**

This readme is located in the project3 folder of the github repository of James Grant Robinett (https://github.com/Jwallee/COE379L-Robinett). Here, we will detail how to use our inference server and test the models avaliable!

---
1. ### **Inference Server and you**

Located in this repository are all of the required files needed to properly run a flask inference server. We have a `Dockerfile`, `api.py` and the `models` folder. Other folders or scrips will not be necessary (unless you want a peek under the hood :0). First, we can containerize our dockerfile using the command
```
docker build -t <username>/proj3-nnet-api .      
```
***Make sure you are in the correct directory! (project3)***

This takes the dockerfile in this project repository and creates and creates the needed docker image! Make sure to replace the `<username>` with your own docker username. The docker container should have copied over `api.py` and the `models` folder.

Next, let's run our docker container using the command:

```
docker run -it --rm -p 5000:5000 <username>/proj3-nnet-api
```

This runs the recently created docker container on port *5000*. Give it some time, it should start up a flask server! Now, we can start looking at whats inside the inference server.

---
2. ### **Whats in the box?!?!**

Now, with the server running, we can make HTTP requests and pass image data to our models! The models contained within this inference server are as follow:
* **An Artificial Neural Network (ANN) version 1**
* **An Artificial Neural Network (ANN) version 2**
* **A LeNet-5 Convolutional Neural Networks (CNN) version 1**
* **An alternate architecture LeNet-5 Convolutional Neural Networks (CNN) version 1**

To make sure you are properly connected to the inference server, open up a new ternimal (so you are not messing with the running flask server) and type the command:
```
curl localhost:5000/
```

This should give a small welcome message, as well as the routes avaliable:
```ruby
{
  "message": "Welcome to the Image Classification API! Here are the available routes:",
  "routes": [
    "/models/best - Best model (Alternative CNN) to classify images",
    "/models/imagesANN/v1 - Dense ANN model",
    "/models/imagesANN/v2 - Dense ANN model 2",
    "/models/imagesCNN/v1 - CNN model",
    "/models/images_altCNN/v1 - Alternative CNN model",
  ]
}
```

You can see from the routes that there are a lot of options. However, the option that matters the most is `/models/best`, which is the best model found during this project (The Alternative CNN). We can get the information about this model using the command:

```
curl localhost:5000/models/best
```
And the output should look something like this:
```ruby
{
  "description": "Classify images that have damage or not using an alternative CNN (Convolutional Neural Network)",
  "name": "images_altCNN (best)",
  "number_of_parameters": 2601153,
  "version": "v1"
}
```
Great! Information about the model is displayed, and all of the other models have a similar format. Give it a try!

---
3. ### **Pass the image please!**

We can now start to send images to the model on the inference server! Here, I will be using a simple python script to make requests and process the image being sent. **Note: The server will only accept singular image files properly formatted. Here is an example on how we prepare an image for an HTTP request:**

```ruby
import requests
import numpy as np
from PIL import Image
import json

# Open the image, resize if needed, convert to numpy array, and rescale
image_path = 'project3Data/damages-split/test/damage/-93.669648_30.220722.jpeg'

img = Image.open(image_path).resize((128, 128))
img_array = np.array(img) / 255.0

# Add the batch dimension and convert to list
img_list = np.expand_dims(img_array, axis=0).tolist()

# The URL of the Flask endpoint, using best model
url = "http://127.0.0.1:5000/models/best"

# Construct the JSON payload with the image data
json_payload = json.dumps({"image": img_list})

# Send the POST request
headers = {'Content-Type': 'application/json'}
response = requests.post(url, data=json_payload, headers=headers)

# Print the response
print("Prediction:", response.json())
```
Running this code, we should get a response looking like this!
```ruby
Prediction: {'result': [[0.05789133161306381]]}
```
This is our damage classification prediction, reported from 0-1! Images classified as `damage` are reported as **0**, and images classified as `no_damage` are reported as **1**. For this result, the model fairly confidenly redicted that the test image is of **damage** classification, and it would be right! Located below is a way to visually see the damage classification, but it is not needed if you know how to interperet the resulting predicted probability.

### *Use this code if you would like to have the prediction be classified as readable 'Damage' or 'No Damage' classifications.*

```ruby
# ... CONTINUING CODE FROM BEFORE
# Send the POST request
headers = {'Content-Type': 'application/json'}
response = requests.post(url, data=json_payload, headers=headers)
print("Prediction:", response.json())

# Check the response
if response.status_code == 200:
    prediction = response.json()['result'][0][0]
    if prediction < 0.5:
        print("Probability <0.5: Damage")
    else:
        print("Probability >=0.5: No Damage")

else:
    print("Error:", response.status_code, response.text)
```

This code gives us a pretty output (ish), and damage is decided on wether or not the predicted probability is above/below 0.5:

```ruby
Prediction: {'result': [[0.05789133161306381]]}
Probability <0.5: Damage
```

And there you have it! You were able to get the best model's metrics and make a prediction by passing an image through a HTTP request and getting a response! You can try the other models by changing the `url="(flaskURL)"` line in the code above using the routes specidied by the welcome message.

---
4. ### **Back for Blood (please be kind to the flask server)**

If you would like, multiple images can be sent using a small code automation we created. Probably not needed, but if you want to send it a bunch of images from both classification and have it count how many it correctly predicts, here you go:

```ruby
# Test the endpoint for the best model
import requests
import numpy as np
from PIL import Image
import json

# I want 50 of 'em
num_images = 50

# Get directory for damage and number of images you want
damage_images = os.listdir('project3Data/damages-split/test/damage')[:num_images]

# Get directory for no_damage and number of images you want
no_damage_images = os.listdir('project3Data/damages-split/test/no_damage')[:num_images]

# The URL of the Flask endpoint
url = "http://127.0.0.1:5000/models/best"

# Keeping count of model classifications
damage_correct = 0
no_damage_correct = 0

# Looping through the damage images
for image_path in damage_images:
    image_path = os.path.join('project3Data/damages-split/test/damage', image_path)
    img = Image.open(image_path).resize((128, 128))
    img_array = np.array(img) / 255.0
    img_list = np.expand_dims(img_array, axis=0).tolist()
    json_payload = json.dumps({"image": img_list})
    response = requests.post(url, data=json_payload, headers=headers)
    if response.status_code == 200:
        prediction = response.json()['result'][0][0]
        if prediction < 0.5:
            damage_correct += 1
    else:
        print("Error:", response.status_code, response.text)

# Looping through the no_damage images
for image_path in no_damage_images:
    image_path = os.path.join('project3Data/damages-split/test/no_damage', image_path)
    img = Image.open(image_path).resize((128, 128))
    img_array = np.array(img) / 255.0
    img_list = np.expand_dims(img_array, axis=0).tolist()
    json_payload = json.dumps({"image": img_list})
    response = requests.post(url, data=json_payload, headers=headers)
    if response.status_code == 200:
        prediction = response.json()['result'][0][0]
        if prediction >= 0.5:
            no_damage_correct += 1
    else:
        print("Error:", response.status_code, response.text)

print("Out of "+str(num_images)+" images each:")
print("Damage Correct:", damage_correct)
print("No Damage Correct:", no_damage_correct)
```
**Output:**
```ruby
Out of 50 images each:
Damage Correct: 49
No Damage Correct: 16
```

And there you go! It counted the number of correct images predicted by the model from the number of images you sent it.