import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time

learning_rate = 0.005
EPOCHS = 15
EARLY_STOP_THRESHOLD = 3
# keeps the data consistent
torch.manual_seed(0)
np.random.seed(0)

valid_ratio = 0.3

print('=> Prepping Data...')
transform = transforms.Compose(
    [
    transforms.RandomRotation(degrees=40),
    transforms.ColorJitter(brightness=0.3, contrast=0.4, hue=0.5, saturation=0.8),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

train_valid_dataset = torchvision.datasets.MNIST(root='./data', 
                                            train=True,
                                            download=True, 
                                            transform=transform)

# test_data = torchvision.datasets.MNIST(root='data',
#                            train=False,
#                            download=True,
#                            transform=transform)

nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset))
nb_valid =  int(valid_ratio * len(train_valid_dataset))
train_dataset, valid_dataset = torch.utils.data.dataset.random_split(train_valid_dataset, [nb_train, nb_valid])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=500,
                                        shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=500,
                                        shuffle=True)
# test_set = torchvision.datasets.MNIST(root='./data', train=False,
#                                        download=True, transform=transform)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
#                                          shuffle=False, num_workers=2)

classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9')

# visualize the MNIST dataset.----------------------------------------------------------------
# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images[:4,]))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

import torch.nn as nn
import torch.nn.functional as F


# construct the CNN------------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1, 16 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# instantiate cnn and print num of parameters ---------------------------------------------------
net = Net()
print(sum([p.numel() for p in net.parameters()]))

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# select device to train the cnn -----------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)

# train CNN and store best model based on validation loss ----------------------------------------
PATH = r"C:\Users\Marc\OneDrive - University of Iowa\College\Machine Learning Project\mnist_net.pth"


start_time = time.time()
best_loss = np.float('inf')
training_losses = []
validation_losses = []
epochs = []

correct = 0
total = 0
percent_correct_array = []
current_time_array = []
test_start_time = time.time()
early_stop_counter = 0
previous_validation_loss = 1000
for epoch in range(EPOCHS):  # loop over the dataset multiple times

    # if early_stop_counter == EARLY_STOP_THRESHOLD:
    #   print("EARLY STOP ENGAGING AT EPOCH: ", epoch)
    #   break

    epochs.append(epoch)
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        

        # print statistics
        running_loss += loss.item()
    epoch_loss = running_loss / (i+1)
    training_losses.append(epoch_loss)
    print("Epoch: ", epoch, " train loss: ", '%.3f' % epoch_loss)
    with torch.no_grad(): 
      running_loss = 0.0
      for i, data in enumerate(valid_loader, 0):
          # get the inputs; data is a list of [inputs, labels]
          inputs, labels = data[0].to(device), data[1].to(device)

          # forward 
          outputs = net(inputs)
          loss = criterion(outputs, labels)

          # validation acc
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

          # print statistics
          running_loss += loss.item()
      epoch_loss = running_loss / (i+1)
      validation_losses.append(epoch_loss)

      # early stopping!
      if epoch_loss > previous_validation_loss:
        early_stop_counter = early_stop_counter + 1
        print('Early stop count', early_stop_counter)

      previous_validation_loss = epoch_loss

      print("Epoch: ", epoch, " validation loss: ", '%.3f' % epoch_loss, ' validation accuracy: ', 100 * correct / total )
      percent_correct_array.append( 100 * correct / total )
      current_time_array.append(time.time() - test_start_time)

      if epoch_loss < best_loss:
        torch.save(net.state_dict(), PATH)
        best_loss = epoch_loss

time_elap = (time.time() - start_time) // 60
print('Finished Training in %d mins' % time_elap)


# getting validation accuracy
correct = 0
total = 0
percent_correct_array = []
current_time_array = []
test_start_time = time.time()
with torch.no_grad():
    for data in valid_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        percent_correct_array.append( 100 * correct / total )
        current_time_array.append(time.time() - test_start_time)

print('Accuracy of the network on the test images: %.3f %%' % (
    100 * correct / total))

# Allows me to graph them in a separate file together
print("current time array")
print(*current_time_array, sep = ", ")
print('validation accuracy')
print(*percent_correct_array, sep = ", ")

print("epochs")
print(*epochs, sep = ", ")
print('training loss')
print(*training_losses, sep = ", ")
print('validation loss')
print(*validation_losses, sep = ", ")

# Plotting training loss at epoch num
plt.plot(epochs, training_losses, marker='o', linestyle='dashed')
plt.xlabel('epoch')
plt.ylabel('training loss')
plt.title('Training Loss vs. Epoch')
plt.show()


# Plots validation accuracy over time
plt.plot(current_time_array, percent_correct_array)
plt.xlabel('Time (s)')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy vs. Time')
plt.show()