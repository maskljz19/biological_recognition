import os
import shutil
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from cv2 import imshow
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms, datasets, models

# 随机种子设置
random_state = 42
np.random.seed(random_state)

# 原始数据集地址
original_dataset_dir = 'D:/python/projects/crawl_images/crawl/last_out'
l = 0
for path in os.listdir(original_dataset_dir):
    l += len(os.listdir(os.path.join(original_dataset_dir, path)))
total_num = int(l / 2)
random_idx = np.array(range(total_num))
np.random.shuffle(random_idx)

# 待处理的数据集地址
base_dir = './dataset'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

# 训练集、测试集的划分
sub_dirs = ['train', 'test']
animals = ['侧柏毒蛾', '八点灰灯蛾', '刺肩普缘蝽', '变色夜蛾', '大叶黄杨尺蛾', '大田鳖'
    , '大皱蝽', '大红蛱蝶', '大青叶蝉', '扁刺蛾', '斑衣蜡蝉', '白星花金龟'
    , '盗毒蛾', '碧凤蝶', '碧蛾蜡蝉', '臭椿皮蛾', '茶翅蝽', '草履蚧'
    , '菜粉蝶', '菜蝽', '薄翅锯天牛', '赤条蝽', '黑蚱蝉']
train_idx = random_idx[:int(total_num * 0.9)]
test_idx = random_idx[int(total_num * 0.9):]
numbers = [train_idx, test_idx]
# for idx, sub_dir in enumerate(sub_dirs):
#     dir = os.path.join(base_dir, sub_dir)
#     if not os.path.exists(dir):
#         os.mkdir(dir)
#     for animal in animals:
#         animal_dir = os.path.join(dir, animal)  #
#         if not os.path.exists(animal_dir):
#             os.mkdir(animal_dir)
#         frompath = 'D:/python/projects/crawl_images/crawl/last_out/' + animal + '/'
#         img_list = os.listdir(frompath)
#         # fnames = [animal[:-1] + '.{}.jpg'.format(i) for i in numbers[idx]]
#         new_numbers = list(map(int,(numbers[idx]/17)))
#         print(new_numbers)
#         for fname in img_list:
#         # for fname in fnames:
#             for number in new_numbers:
#                 if str(int(number)) in fname:
#                     src = os.path.join(os.path.join(original_dataset_dir,animal), fname)
#                     dst = os.path.join(animal_dir, animal+fname)
#                     shutil.copyfile(src, dst)
#
#         # 验证训练集、验证集、测试集的划分的照片数目
#         print(animal_dir + ' total images : %d' % (len(os.listdir(animal_dir))))
# coding=utf-8

# 配置参数
random_state = 1
torch.manual_seed(random_state)  # 设置随机数种子，确保结果可重复
torch.cuda.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
np.random.seed(random_state)
# random.seed(random_state)

epochs = 10  # 训练次数
batch_size = 4  # 批处理大小
num_workers = 0  # 多线程的数目
use_gpu = torch.cuda.is_available()
model_name = "model.pt"
PATH = './'+model_name
# 对加载的图像作归一化处理， 并裁剪为[84x84x3]大小的图像
data_transform = transforms.Compose([
    transforms.Resize(84),
    transforms.CenterCrop(84),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root='./dataset/train_dataset/',
                                     transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers)

test_dataset = datasets.ImageFolder(root='./dataset/test_dataset/', transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


# 创建模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 18 * 18, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 23)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 18 * 18)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


net = Net()
if (os.path.exists(model_name)):
    # net = torch.load('model.pt')
    net = torch.load(model_name, map_location='cpu')

if use_gpu:
    net = net.cuda()
print(net)

# 定义loss交叉熵损失函数和optimizer
cirterion = nn.CrossEntropyLoss()
# 带momentum动量的SGD优化
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)


def train():
    for epoch in range(epochs):
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, train_labels = data
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(train_labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(train_labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            _, train_predicted = torch.max(outputs.data, 1)
            train_correct += (train_predicted == labels.data).sum()
            loss = cirterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_total += train_labels.size(0)

        print('train %d epoch loss: %.3f  acc: %.3f ' % (
            epoch + 1, running_loss / train_total, 100 * train_correct / train_total))

    torch.save(net, 'model.pt')


def test():
    # 模型测试
    correct = 0
    test_loss = 0.0
    test_total = 0
    test_total = 0
    net.eval()
    for data in test_loader:
        images, labels = data
        if use_gpu:
            images, labels = Variable(images.cuda()), Variable(labels.cuda())
        else:
            images, labels = Variable(images), Variable(labels)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        loss = cirterion(outputs, labels)
        test_loss += loss.item()
        test_total += labels.size(0)
        correct += (predicted == labels.data).sum()
        print('预测值：' + ' '.join('%5s' % animals[predicted[j]] for j in range(len(predicted))))
        print('实际值：' + ' '.join('%5s' % animals[labels[j]] for j in range(len(predicted))))

    print('test  %d epoch loss: %.3f  acc: %.3f ' % (1, test_loss / test_total, 100 * correct / test_total))


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def my_test():
    # 模型测试
    test_dataset = datasets.ImageFolder(root='./dataset/mytest/', transform=data_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers)
    correct = 0
    test_loss = 0.0
    test_total = 0
    test_total = 0
    net.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            if use_gpu:
                images, labels = Variable(images.cuda()), Variable(labels.cuda())
            else:
                images, labels = Variable(images), Variable(labels)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = cirterion(outputs, labels)
            test_loss += loss.item()
            test_total += labels.size(0)
            correct += (predicted == labels.data).sum()
            for j in range(len(predicted)):
                imshow(torchvision.utils.make_grid(images[j]))
            print('预测值：' + ' '.join('%5s' % animals[predicted[j]] for j in range(len(predicted))))
            print('实际值：' + ' '.join('%5s' % animals[labels[j]] for j in range(len(predicted))))

    print('test  %d epoch loss: %.3f  acc: %.3f ' % (1, test_loss / test_total, 100 * correct / test_total))


# train()
my_test()
# test()