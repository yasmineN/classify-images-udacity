# Import libraries
import argparse
import torch
from torchvision import datasets, transforms, models
from collections import OrderedDict
from os.path import isdir
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import time
import sys
import platform



def get_input_args():
    """
    Retrieves and parses the 8 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 8 command line arguments. If 
    the user fails to provide some or all of the arguments, then the default 
    values are used for the missing arguments. 
    
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """


    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create command line arguments as mentioned above using add_argument() from ArguementParser method
    ## Argument 1: that's a path to a folder
    parser.add_argument('--data_dir', type = str, default = 'flowers/', 
                    help = 'path to the folder of flowers') 
    
    ## Argument 2: The CNN model architecture to use
    parser.add_argument('--arch', type = str, default = 'vgg16', choices = ['vgg16', 'alexnet', 'resnet'],
                    help = 'CNN model architecture to use') 

    ## Argument 3: that's a path to a folder to save checkpoints
    parser.add_argument('--save_dir', type = str, default = 'checkpoints', 
                    help='Directory to save checkpoints') 

    ## Argument 4: Learning rate 
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                    help='Learning rate')

    ## Argument 5: Hidden units
    parser.add_argument('--hidden_units', type=int, default=256, 
                    help='hidden units')

    ##Argument 6: Epochs
    parser.add_argument('--epochs', type=int, default=5, help='Epoch count')

    ## Argument 7: Use GPU
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='Use GPU for training')

    # Set-defaults
    parser.set_defaults(gpu=False)

    in_args = parser.parse_args()

    return in_args


def check_device(gpu_arg):
   # If gpu_arg is false then simply return the cpu device
    has_mps = getattr(torch,'torch.backends.mps.is_built()',False) # check for MPS (Apple Metal) / MAC with m1
    has_gpu = torch.cuda.is_available()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    if gpu_arg=='gpu' and has_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"Target device is {device} ")
    return device



def load_data(data_dir='flowers'):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    # Define your transforms for the training, validation, and testing sets
    train_transform = transforms.Compose([transforms.RandomRotation(degrees=30),
                                          transforms.RandomResizedCrop(size=(224)),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    valid_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                              [0.229, 0.224, 0.225])]) 

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=train_transform)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

    data = {'train_data': train_data, 
            'valid_data': valid_data, 
            'test_data': test_data, 
            'train_loader': train_loader, 
            'valid_loader': valid_loader, 
            'test_loader': test_loader}

    return data




def load_model(device, arch, hidden_units=256, learning_rate=0.001 ):

   
    if arch == "vgg16":
        model = models.vgg16(weights='VGG16_Weights.DEFAULT')
        weights='VGG16_Weights.DEFAULT'
        num_in_features = 25088
        hidden_units = 256
        hidden_units2 = 128

    elif arch == 'alexnet':
        model = models.alexnet(weights='AlexNet_Weights.IMAGENET1K_V1')
        weights='AlexNet_Weights.IMAGENET1K_V1'
        num_in_features = 9216
        hidden_units = 4096
        hidden_units2 = 2048

    elif arch == 'resnet':
        model = models.resnet152(weights='ResNet152_Weights.DEFAULT')
        weights='ResNet152_Weights.DEFAULT'
        num_in_features = 2048
        hidden_units = 1000
        hidden_units2 = 500

    else:
         print(f"Please choose one of these networks: vgg, alexnet, or resnet")
         return

    

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Initialize classifier
    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(num_in_features, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, hidden_units2)),
        ('relu', nn.ReLU()),
        ('fc3', nn.Linear(hidden_units2, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))


    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)

    model_info = {'model': model,
                  'criterion': criterion,
                  'optimizer':optimizer,
                  'weights': weights
                  }
    print(f"Training using {arch} network ")

    return model_info



def train_model(model, trainloader, testloader, optimizer, criterion,  device, epochs=5, print_every=10):
    train_loss = 0
    steps = 0
    running_loss = 0
    
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                

                model.train()




def validate_model(test_loader):
    accuracy = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            output = model.forward(inputs)
            
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            
    print(f'Test accuracy: {accuracy/len(test_loader):.3f}')



def save_checkpoint(arch, weights, model, optimizer, train_data, save_path ):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = { 
               'arch': 'vgg16',
               'weights':'VGG16_Weights.DEFAULT',
               'classifier': model.classifier,
               'state_dict': model.state_dict(),
               'optimizer': optimizer.state_dict(),
               'class_to_idx': train_data.class_to_idx,
             }

    torch.save(checkpoint, save_path)



def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(weights= 'VGG16_Weights.DEFAULT')
    
    # freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = checkpoint['classifier']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
      
    return model

def main():

    in_args =  get_input_args()

    data_dir = in_args.data_dir
    save_dir = in_args.save_dir
    learning_rate = in_args.learning_rate
    hidden_units = in_args.hidden_units
    epochs = in_args.epochs
    gpu = in_args.gpu
    arch = in_args.arch


    #set the path - transform - loader
    data= load_data(data_dir)
    
    # device
    device = check_device(gpu)


    # #returns model info 
    model_info = load_model(device, arch, hidden_units, learning_rate)

    # set timer - before training
    start_time = time.time()


    #train 
    train_model(model_info['model'], 
                data['train_loader'], 
                data['valid_loader'], 
                model_info['optimizer'], 
                model_info['criterion'], 
                device, 
                epochs, 
                print_every=20 )


    # save time - after training is done
    end_time = time.time()

    # save the model - load it later for making predictions.  
    save_checkpoint(arch,
                    model_info['weights'],
                    model_info['model'], 
                    model_info['optimizer'], 
                    data['train_data'], 
                    save_path='checkpoint.pth')

    #Computes training runtime 
    tot_time = end_time - start_time #calculate difference between end time and start time
    print("\n** Training time:",
          str(round((tot_time/3600)))+":"+str(round((tot_time%3600)/60))+":"
          +str(round((tot_time%3600)%60)) )


if __name__ == '__main__':
    main()
