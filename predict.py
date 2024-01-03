import argparse
import torch
from torchvision import transforms, models
import torch.nn.functional as F
import numpy as np
from PIL import Image
import json
import os


from train import load_checkpoint


def get_input_args():

    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create command line arguments as mentioned above using add_argument() from ArguementParser method

    ## Argument 1: path to the flower to test predict
    parser.add_argument('--input', type = str, default = 'flowers/test/3/image_06641.jpg', 
                    help='path to the flower to test predict') 

    ## Argument 2: that's the checkpoint file
    parser.add_argument('checkpoint',  default='checkpoint.pth', help="checkpoint file")

    
    ## Argument 3: Set the number of n top categories to return
    parser.add_argument('--top_k', type = int, default = 5, help="Number of top categoryies") 
    
    ## Argument 4: file that contains the list of valid flower categories
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', 
                    help = 'list of valid flowers') 

    # GPU
   # parser.add_argument('--gpu', dest='gpu', action='store_true', help='Use GPU for training')

    # Set-defaults
    parser.set_defaults(gpu=False)


    in_args = parser.parse_args()

    return in_args



def load_category_names(category_names = 'cat_to_name.json'):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name




def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                              [0.229, 0.224, 0.225])])
    
    image = img_transforms(img)
    
    return image

def predict(image_path, model, top_k=5, device ='mps'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file


    # if device == 'gpu':
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")

    model.eval()
    model.to(device)



    input_img = process_image(image_path)
    input_img = input_img.unsqueeze_(0)
    input_img = input_img.float()
    
    with torch.no_grad():
        output = model(input_img.to(device))
        ps = torch.exp(output)
        top_ps, top_classes = ps.topk(top_k, dim=1)
        
        top_ps = top_ps.tolist()[0]
        top_classes = top_classes.tolist()[0]

        idx_to_class = {value:key for key, value in model.class_to_idx.items()}
        top_classes = [idx_to_class.get(x) for x in top_classes]
        
    return top_ps, top_classes


def main(): 
    in_args =  get_input_args()


    device = in_args.gpu
    file_path = in_args.input
    top_k = in_args.top_k
    category_names = in_args.category_names
    checkpoint = in_args.checkpoint
    device = 'mps'


    model = load_checkpoint(checkpoint)
    cat_to_name = load_category_names(category_names)

    probs, classes = predict(file_path, model, top_k , device)

    flower_class = [cat_to_name.get(flower) for flower in classes] 
    flower_name = flower_class[0]
    print(flower_name)
    
    i=0 # 
    while i < len(flower_class):
        print("probability of {} is {}".format(flower_class[i], probs[i]))
        i += 1 # cycle through

if __name__ == "__main__":
    main()