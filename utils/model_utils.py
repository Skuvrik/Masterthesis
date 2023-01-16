import torch
from torch import nn
import numpy as np
import pydicom as pdc

def train(train_loader, model, criterion, optimizer, device):
    losses = 0
    torch.cuda.empty_cache()
    model.train()
    for i, (input,target) in enumerate(train_loader):
        target = torch.unsqueeze(target,1)
        target = target.to(device)
        input = torch.permute(torch.unsqueeze(input, 0), (1, 0, 2, 3))
        input = input.to(device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss / len(target)
    return losses

def validate(test_loader, model, criterion, device, apr):
    torch.cuda.empty_cache()
    model.eval()
    losses = 0
    TP, FP, TN, FN = 0, 0, 0, 0
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            target = torch.unsqueeze(target,1)
            target = target.to(device)
            input = torch.permute(torch.unsqueeze(input, 0), (1, 0, 2, 3))
            input = input.to(device)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            output = model(input_var)
            if apr:
                for actual, predicted in zip(target, output):
                    predicted = int(round(torch.sigmoid(predicted).item()))
                    actual = int(actual.item())
                    if predicted == 1 and actual == 1:
                        TP += 1
                    elif predicted == 1 and actual == 0:
                        FP += 1
                    elif predicted == 0 and actual == 0:
                        TN += 1
                    elif predicted == 0 and actual == 1:
                        FN += 1
                    else:
                        pass
            loss = criterion(output, target_var)
            losses += loss / len(target)
        if apr:
            try:
                accuracy = round((TP+TN)/(TP+FP+TN+FN),2)
            except:
                accuracy = 0
            try:
                precision = round(TP/(FP+TP), 2)
            except:
                precision = 0
            try: 
                recall = round(TP/(TP+FN), 2)
            except:
                recall = 0
            print(f"TP {TP}, FP {FP}, TN {TN}, FN {FN}")
            return [losses, accuracy, precision, recall]
    return [losses]

def train_and_eval(model, test_loader, train_loader, epochs, device, apr=True):
    losses = np.zeros((epochs, 2))
    metrics = np.zeros((epochs, 3))
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)
    for epoch in range(epochs):
        print(f"Epoch {epoch +1} of {epochs}")
        train_loss = train(train_loader, model, criterion, optimizer, device)
        val_loss = validate(test_loader, model, criterion, device, apr)
        losses[epoch,0], losses[epoch,1] = train_loss.item(), val_loss[0].item()
        if apr:
            metrics[epoch,0], metrics[epoch,1], metrics[epoch,2] = val_loss[1], val_loss[2], val_loss[3]
            print(f"Validation - Accuracy {metrics[epoch,0]} Precision {metrics[epoch, 1]} and Recall {metrics[epoch, 2]}")
        print(f"Training loss {losses[epoch, 0]}, validation loss {losses[epoch,1]}\n")
    return [losses, metrics]

def get_model_performance_metrics(model, images, labels, device, normalize):
    # Copy dataframe due to the use of iterrows
    images_copy = images.copy()
    TP_list, FP_list, TN_list, FN_list = [], [], [], []
    for _, selected in images_copy.iterrows():
        model.eval()
        image = pdc.read_file(selected['ImagePath'])
        image = np.expand_dims(image.pixel_array.astype("int16"), axis=(0,1))
        if normalize: 
            image = image / np.max(image)
        image = torch.tensor(image, dtype=torch.float)
        image = image.to(device)
        actual = labels[(labels['Patient'] == int(selected['Patient'])) & (labels['Slice'] == int(selected['Slice']))]['Label'].item()
        predicted = int(round(torch.sigmoid(model(image)).item()))
        if predicted == 1 and actual == 1:
            TP_list.append(selected)
        elif predicted == 1 and actual == 0:
            FP_list.append(selected)
        elif predicted == 0 and actual == 0:
            TN_list.append(selected)
        elif predicted == 0 and actual == 1:
            FN_list.append(selected)
        else:
            pass
    TP, FP, TN, FN = len(TP_list), len(FP_list), len(TN_list), len(FN_list)
    try:
        accuracy = round((TP+TN)/(TP+FP+TN+FN),2)
    except:
        accuracy = 0
    try:
        precision = round(TP/(FP+TP), 2)
    except:
        precision = 0
    try: 
        recall = round(TP/(TP+FN), 2)
    except:
        recall = 0
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")
    return [TP_list, FP_list, TN_list, FN_list]