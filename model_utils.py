import torch
from torch import nn

def train(train_loader, model, criterion, optimizer, device):
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

def validate(test_loader, model, criterion, device):
    torch.cuda.empty_cache()
    model.eval()
    losses = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            target = torch.unsqueeze(target,1)
            target = target.to(device)
            input = torch.permute(torch.unsqueeze(input, 0), (1, 0, 2, 3))
            input = input.to(device)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            loss = criterion(output, target_var)
            losses += loss / len(target)
    return losses

def train_and_eval(model, test_loader, train_loader, epochs, device):
    losses = []
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)
    for epoch in range(epochs):
        print(f"Epoch {epoch +1} of {epochs}")
        train(train_loader, model, criterion, optimizer, device)
        loss = validate(test_loader, model, criterion, device)
        losses.append(loss)
        print(f"Losses: {loss}")

    return losses