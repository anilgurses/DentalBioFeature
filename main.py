import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import pandas as pd
import torchvision
import cv2
from torchvision import transforms, utils, models
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.models as models
from os import walk
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime
from Tooth_Dataset import ToothDataset
from data_load import tooth_data_load, other_tooth_data_load


learning_rate = 0.001
num_epochs = 30
num_classes = 2

writer = SummaryWriter('log')


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    last_lay = ""
    #nets = ["alexnet", "resnet", "vgg", "inception" , "squeezenet", "densenet"]

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        last_lay = "layer4"

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        last_lay = "features"

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        last_lay = "features"
    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224
        last_lay = "features"
    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        last_lay = "features"
    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299
        last_lay = "Mixed_7c"
    elif model_name == "mobilenet_v2":
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        last_lay = "features"
    elif model_name == "googlenet":
        model_ft = models.googlenet(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        last_lay = "aux2"
    elif model_name == "shufflenet":
        model_ft = models.shufflenet_v2_x1_0(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        last_lay = "conv5"

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft,last_lay

def train(train_data,model,modelname):
    net = model.cuda()
    print("-- Network Retrieved --")
    optimizer = torch.optim.SGD(net.parameters(), learning_rate,  momentum=0.9, weight_decay=1e-2, nesterov=True)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)

    criterion = nn.CrossEntropyLoss().cuda()
    net.train()
    print("-- Training initialized --")
    stp = 0
    for epoch in range(num_epochs):
        accuracy = 0
        losses = 0

        for i_batch, data_batched in enumerate(train_data):
            img = data_batched["image"].cuda()
            clss = data_batched["other"].cuda()
            
            output = net(img).cuda()

            optimizer.zero_grad()
            loss = criterion(output, clss)
            
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            stp += 1

            with torch.no_grad():
                _, preds = torch.max(output, 1)

                accuracy += preds.eq(clss.view_as(preds)).sum().item()
                losses += loss.item()
                if i_batch % 20 == 0 and i_batch != 0:
                #print('-- Epoch: {} \tStep: {}/{} \tLoss: {:.4f} \tAcc: %{:.4f}'.format(epoch + 1, i_batch,total_steps, losses/10, accuracy.item()/(10*len(clss.squeeze()))))
                    print('{{"metric": "loss", "value": {}, "step":{},"epoch":{}}}'.format(
                        losses/20, stp, epoch))
                    print('{{"metric": "accuracy", "value": {}}}'.format(
                        accuracy/(20*len(clss))))
                    writer.add_scalar('training loss',
                                        losses/20,
                                        stp)

                    writer.add_scalar('accuracy',
                                        accuracy/(20*len(clss)),
                                        stp)
                    accuracy = 0
                    losses = 0

        torch.save(net.state_dict(), modelname +'-'+'model-00' + repr(epoch % 5) + '.pth')

        
    return net

def validation(net,valdata,modelname):
  net.eval()
  test_loss = 0
  correct = 0
  net.cuda()
  correct = 0
  total = 0
  items = {"Real":[],"Predicted":[],"Prob":[]}
  with torch.no_grad():
    for i_batch, data_batched in enumerate(valdata):
      img = data_batched["image"].to("cuda")
      clss = data_batched["other"].to("cuda")
      output = net(img).to("cuda")
      print(clss)
      pred_prob, predicted = torch.max(output, 1)
      print(predicted)
      #test_loss += F.nll_loss(torch.exp(output), clss.squeeze(1), reduction='sum').item() 
      items["Real"].extend(clss.squeeze())
      items["Predicted"].extend(predicted)
      items["Prob"].extend(pred_prob)

      correct += predicted.eq(clss.squeeze().view_as(predicted)).sum().item()
  #test_loss  /= len(tsne_load.dataset)
  print('\n Accuracy: {}/{} ({:.0f}%)\n'.format(
      correct, len(valdata.dataset),
      100. * correct / len(valdata.dataset)))

  now = datetime.now()
  date_time = now.strftime("%d-%m-%H:%M")
  df = pd.DataFrame(
      items, columns=['Real', 'Predicted', 'Prob'])
  path = 'outputs/' + modelname+'_' + date_time + '_score.csv'
  df.to_csv(path, index=False)


def extract_features(model,val_data,last_lay):
   
    print("**** Train is finished ****")

    model.eval()

    features = []
    ids = []
    tooths = []
    def hook(module, input, output):
            N,C,H,W = output.shape
            output = output.reshape(N,C,-1)
            features.append(output.mean(dim=2).cpu().detach().numpy())
    handle = model._modules.get(last_lay).register_forward_hook(hook)
    
    for i_batch, inputs in tqdm(enumerate(val_data), total=len(val_data)):
            _ = model(inputs["image"].to("cuda").cuda())
            ids.extend(inputs["personid"])
            tooths.extend(inputs["toothid"])

    features = np.concatenate(features)
    features = pd.DataFrame(features)
    features = features.add_prefix('Feature_')
    
    ids = pd.DataFrame(ids)
    tooths = pd.DataFrame(tooths)
    features['Id'] = ids
    features['ToothId'] = tooths

    handle.remove()
    del model
    return features,ids,tooths



def eval_frozen(modelpath):
  tooth_load = other_tooth_data_load()
  net = model.Alexnet(pretrained=False)
  net.load_state_dict(torch.load(
        modelpath, map_location=torch.device('cuda')))

  #eval_model_single(net, tooth_load)





if __name__ == '__main__':
      

  parser = argparse.ArgumentParser(description='Tooth NETWORK TRAINER AND TESTER')
  parser.add_argument("--path",  default="model4_multi.pth", type=str, help="Frozen model path")
  parser.add_argument("--mode", default="train",type=str, help="eval or train")
  parser.add_argument("--lr", default=0.001, type=float, help="Learning Rate")
  parser.add_argument("--ne", default=100,type=int, help="Number of Epochs")
  parser.add_argument("--ft", default="Fine", type=str, help="Fine Tuning")
  args = parser.parse_args()

  
  learning_rate = args.lr
 
  num_epochs = args.ne
  ft = args.ft

  print("-- Sequence Started --")
 
  train_data, val_data = tooth_data_load()

  #feats = extract_features("Alexnet",train_data)
  #, "alexnet", ,
  nets = ["alexnet","resnet", "squeezenet", "densenet","mobilenet_v2", "shufflenet"]


  for nt in nets:
   
    

    if ft == "Fine":
        model,lastlay = initialize_model(nt, num_classes, False, False)

        model = model.cuda()
        print("********"+ nt + " ***Training is started*********")
        net = train(train_data, model,nt)

        validation(net,val_data, nt)
        
        feats, ids, tooths  = extract_features(net,val_data,lastlay)

        feats.to_csv(nt+"_"+"feats.csv")
        ids.to_csv(nt+"_"+"ids.csv")
        tooths.to_csv(nt+"_"+"tooths.csv")
    else:
        print("********* FineTune on proccess")
        model,lastlay = initialize_model(nt, num_classes, False, True)

        net = model.cuda()

        feats, ids, tooths  = extract_features(net,val_data,lastlay)

        feats.to_csv(nt+"_"+"feats_withoudfine.csv")
        ids.to_csv(nt+"_"+"ids_withoudfine.csv")
        tooths.to_csv(nt+"_"+"tooths_withoudfine.csv")
