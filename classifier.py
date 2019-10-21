from torchvision import models
from workspace_utils import active_session
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import json
from classifier_utils import process_image

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)

def generate_model_components(arch, lr, h_units, categories):
    # set up base model
    model = None
    classifier_input = None
    optimizer = None
    
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad = False
        
        classifier_input = 25088
        classifier = Network(classifier_input, len(categories), h_units)
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    elif arch == "resnet34":
        model = models.resnet34(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad = False
        
        classifier_input = 512
        classifier = Network(classifier_input, len(categories), h_units)
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    else:
        raise Exception("model architecture not supported")
    
    # define criteron and optimizer
    criterion = nn.NLLLoss()
    return model, criterion, optimizer
    
def train_model(model, optimizer, criterion, train_loader, validation_loader, epochs=12, gpu=False):
    # set up device
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    # move model to whichever device is available, prioritizing GPU
    model.to(device)
    
    # train network
    steps = 0
    print_every = 10

    with active_session():
        for epoch in range(epochs):
            running_loss = 0
            for images, labels in train_loader:
                steps += 1
                # move training data to same device as model
                images , labels = images.to(device), labels.to(device)

                # train loop
                optimizer.zero_grad()
                logprbs = model.forward(images)
                loss = criterion(logprbs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # print validation performance every few steps
                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    # go into eval mode: turns off dropout to have full power when making predictions
                    model.eval() 

                    for inputs2, labels2 in validation_loader:
                        optimizer.zero_grad()
                        # move test data to same device as model
                        inputs2, labels2 = inputs2.to(device), labels2.to(device)

                        with torch.no_grad():
                            logps = model.forward(inputs2)
                            test_loss = criterion(logps, labels2)

                            # calc accuracy
                            # model returns log probs, so just use torch.exp to get actual probabilities for each class
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equality = top_class == labels2.view(*top_class.shape)
                            accuracy += torch.mean(equality.type(torch.FloatTensor)).item()


                    test_loss = test_loss / len(validation_loader)
                    accuracy = accuracy / len(validation_loader)

                    print(f"Epoch: {epoch+1}/{epochs}"
                         f"Train loss: {running_loss / print_every:.3f}..."
                         f"Validation Loss: {test_loss:.3f}..."
                         f"Validation Accuracy: {accuracy:.3f}...")


                    running_loss = 0
                    model.train() # put model back into train mode to activate dropout
                    
def eval_model(model, criterion, test_loader, gpu=False):
    # set up device
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    # Do validation on the test set
    test_loss = 0
    accuracy = 0
    # go into eval mode: turns off dropout to have full power when making predictions
    model.eval() 

    with torch.no_grad():
        for inputs, labels in test_loader:
            # move test data to same device as model
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
            test_loss = criterion(logps, labels)

            # calc accuracy
            # model returns log probs, so just use torch.exp to get actual probabilities for each class
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test Loss: {test_loss / len(test_loader):.3f}..."
         f"Test Accuracy: {accuracy / len(test_loader):.3f}...")
    
def save_model(model, h_units, categories, class_to_idx, filepath="checkpoint.pth", arch="vgg16"):
    model.class_to_idx = class_to_idx
    model.cpu()
    
    archs = {
        "vgg16": 25088,
        "resnet34": 512
    }
    
    # Save the checkpoint
    checkpoint = {
        'structure': arch,
        'input_size': archs[arch],
        'output_size': len(categories),
        'hidden_layers': h_units,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict()
    }

    torch.save(checkpoint, filepath)
    print("model saved")
    
def load_model(filepath, categories):
    
    archs = {
        "vgg16": 25088,
        "resnet34": 512
    }
    
    checkpoint = torch.load(filepath)
    
    if checkpoint["structure"] == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier = Network(archs[checkpoint["structure"]], len(categories), checkpoint["hidden_layers"])
    elif checkpoint["structure"] == "resnet34":
        model = models.resnet34(pretrained=True)
        model.fc = Network(archs[checkpoint["arch"]], len(categories), checkpoint["hidden_layers"])
    else:
        raise Exception("Model architecture not supported")
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint["state_dict"])
    for param in model.parameters(): param.requires_grad = False
    print("model loaded")
    return model

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    
    with torch.no_grad():
        im = process_image(image_path)
        im.unsqueeze_(0) # adds 1 to image dimensions to match up with what model expects
        output = model(im)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk)
        return top_p, top_class