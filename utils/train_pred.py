import torch
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt

def train(model, num_epochs, train_loader):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    loss_history = []
    acc_history = []
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # data = data.to(device)
            # target = target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.to(torch.float32))
            loss.backward()
            optimizer.step()
            accuracy = (output.argmax(dim=1) == target.argmax(dim=1)).float().mean().to('cpu').numpy()

        if epoch == num_epochs-1:
            print(output)
            print(target)
                
        loss_history.append(loss.item())
        acc_history.append(accuracy)
    print('Finished Training')
    plt.plot(loss_history)
    plt.plot(acc_history)

def test(model, test_loader):
    correct = 0
    all_pred_labels = torch.tensor([])
    all_true_labels = torch.tensor([])
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred_labels = output.argmax(dim=1, keepdim=True)
            true_labels = target.argmax(dim=1, keepdim=True)
            all_pred_labels = torch.cat((all_pred_labels, pred_labels.to('cpu')), dim=0)
            all_true_labels = torch.cat((all_true_labels, true_labels.to('cpu')), dim=0)
            correct += pred_labels.eq(true_labels).sum().item()
 
    return correct / len(test_loader.dataset), all_pred_labels.numpy().tolist(), all_true_labels.numpy().tolist()
