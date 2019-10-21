from classifier import *
import argparse
import json
import numpy as np

parser = argparse.ArgumentParser(
    description='CLI for making predictions with trained a neural network to classify images',
)

parser.add_argument('image_path', action="store")
parser.add_argument('saved_model', action="store")
parser.add_argument('--top_k', action="store", dest="topk", default=1, type=int, help="highest probabilities and their respective classes")
parser.add_argument('--gpu', action='store_true', default=False, dest='gpu_train', help='True if model should be trained on a cuda GPU')
parser.add_argument('--categories', action='store', dest='categories', default="cat_to_name.json", help="filepath to json file containing category mappings")

results = parser.parse_args()

print("loading category mappings")
with open(results.categories, 'r') as f:
    cat_names = json.load(f)

print("Loading model")
model = load_model(results.saved_model, cat_names)

print("making prediction")
probs, classes = predict(results.image_path, model)
cats = [cat_names[str(index)] for index in np.array(classes[0])]
print(f"Probablities: {probs}")
print(f"Classes: {classes}")
print(f"Category Names: {cats}")

