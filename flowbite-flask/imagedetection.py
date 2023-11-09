# run this file and look at output in terminal. We need to find a way
# to determine if the snake that is detected is venomous or not
# add add all these files to the website
from imageai.Classification.Custom import CustomImageClassification
import os
from torchvision.models import resnet50, ResNet50_Weights

execution_path = os.getcwd()

# Using the resnet-50 model
prediction = ImageClassification() # create an instance of the ImageClassification class from the ImageAI library
prediction.setModelTypeAsResNet50() # set the model type that the ImageClassification object will use to ResNet-50
#prediction.setModelPath(os.path.join(execution_path, "resnet50-19c8e357.pth")) # set the path to the model weights file that the ImageClassification object will use

prediction.setModelPath(os.path.join(execution_path, "resnet50-snakes-test_acc_0.23901_epoch-48.pt"))
prediction.setJsonPath(os.path.join(execution_path, "snakes_model_classes.json"))

prediction.loadModel() # load the model with the specified model type (ResNet-50) and model weights (from the included .pth file)

def image_recognition(image):
    print(image)
    predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, image), result_count=5) # call the classifyImage method of the ImageClassification object to classify the snake in the image
    for eachPrediction, eachProbability in zip(predictions, probabilities): # Iterates through the top 5 predictions in order of decreasing probability
        print(eachPrediction , " : " , eachProbability)
    predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, image), result_count=1)
    print(f"\n\nThis snake is: {predictions[0]}\n\n")
    return predictions[0]
