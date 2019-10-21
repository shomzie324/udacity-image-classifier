from classifier import *
from classifier_utils import get_data_loaders
import argparse

parser = argparse.ArgumentParser(
    description='CLI for training a neural network to classify images',
)

parser.add_argument('data_dir', action="store")
parser.add_argument('--save_dir', action="store", dest="save_dir", default="checkpoint.pth", help="where to save the trained model")

parser.add_argument('--arch', action="store", dest="arch", default="vgg16", help="which base CNN to use")
parser.add_argument('--learning_rate', action="store", dest="learning_rate", default=0.001, type=int, help="how much to updated weights on each back prop")
parser.add_argument('--hidden_units', action='append', dest='hidden_units', default=[300,150], type=int, help='number of units in each hidden layer')

parser.add_argument('--epochs', action="store", dest="epochs", default=12, type=int, help="how many times model should train on the dataset")
parser.add_argument('--gpu', action='store_true', default=False, dest='gpu_train', help='True if model should be trained on a cuda GPU')
parser.add_argument('--categories', action='store', dest='categories', default="cat_to_name.json", help="filepath to json file containing category mappings")

results = parser.parse_args()

print("getting split data")
train_loader, validation_loader, test_loader, class_to_idx = get_data_loaders(results.data_dir)

print("preparing model components")
with open(results.categories, 'r') as f:
    cat_names = json.load(f)
model, criterion, optimizer = generate_model_components(results.arch, results.learning_rate, results.hidden_units, cat_names)

print("training model")
train_model(model, optimizer, criterion, train_loader, validation_loader, epochs=results.epochs, gpu=results.gpu_train)

print("evaluating model")
eval_model(model, criterion, test_loader, gpu=results.gpu_train)

print("saving model")
save_model(model, results.hidden_units, cat_names, class_to_idx, arch="vgg16", filepath=results.save_dir)


