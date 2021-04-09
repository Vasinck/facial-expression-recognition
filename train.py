from PIL.Image import Image
import numpy as np
import torchvision.models as models
import torch.utils.data as data
from torchvision import transforms
import cv2
import pandas as pd
import os ,torch
import torch.nn as nn
import random
import deal_with_photo
import torch.nn.functional as F

class basic_parameters:
    def __init__(self):
        self.raf_path = 'g:/python/Datasets/raf-basic/basic/'
        self.beta = 0.7
        self.relabel_epoch = 10
        self.margin_1 = 0.15
        self.margin_2 = 0.2 
        self.batch_size = 64
        self.lr = 0.01
        self.momentum = 0.9
        self.epochs = 70
        self.drop_rate = 0.3
        self.pretrained = False

class raf_datasets(data.Dataset):
    def __init__(self, raf_path, phase, transform = None, basic_aug = False):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
        if phase == 'train':
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]
        file_names = dataset.iloc[:, NAME_COLUMN].values
        # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        self.label = dataset.iloc[:, LABEL_COLUMN].values - 1 
        
        self.file_paths = []
        for f in file_names:
            f = f.split(".")[0]
            f = f +"_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image', f)
            self.file_paths.append(path)
        
        self.basic_aug = basic_aug
        self.aug_func = [deal_with_photo.flip_image,deal_with_photo.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]
        label = self.label[idx]
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0,1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label, idx

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)


class attention2d(nn.Module):
    def __init__(self, in_planes, k):
        super(attention2d, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, k, 1)
        self.fc2 = nn.Conv2d(k, k, 1)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.view(x.size(0), -1)
        x = F.softmax(x)
        return x


class dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4):
        super(dynamic_conv2d, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, K)

        self.weight = nn.Parameter(torch.Tensor(K, out_planes, in_planes//groups,
                                                kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None

    def forward(self, x): 
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)
        weight = self.weight.view(self.K, -1)

        aggregate_weight = torch.mm(softmax_attention, weight).view(-1,
                                                                    self.in_planes, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output

class basic_net(nn.Module):
    def __init__(self, pretrained = True, drop_rate = 0):
        super(basic_net, self).__init__()
        self.drop_rate = drop_rate
        resnet  = models.resnet18(pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc_in_dim = list(resnet.children())[-1].in_features
        self.features = self.features[: -1]

    def forward(self, x):
        x = self.features(x)
        if self.drop_rate > 0:
            x =  nn.Dropout(self.drop_rate)(x)
        return x

class attention_net(nn.Module):
    def __init__(self, conv_in_dim = 512):
        super(attention_net, self).__init__()
        self.faltten = FlattenLayer()
        self.global_avg_pool = GlobalAvgPool2d()
        
        self.conv1_1 = dynamic_conv2d(conv_in_dim, conv_in_dim // 2, kernel_size=3, padding=1, stride=1)
        self.conv1_2 = dynamic_conv2d(conv_in_dim // 2, conv_in_dim // 4, kernel_size=3, padding=1, stride=1)
        self.bn1_1 = nn.BatchNorm2d(conv_in_dim // 2)
        self.bn1_2 = nn.BatchNorm2d(conv_in_dim // 4)
        self.conv1_3 = dynamic_conv2d(conv_in_dim // 4, conv_in_dim // 4, kernel_size=8, padding=1)
        self.conv1_4 = dynamic_conv2d(conv_in_dim // 4, conv_in_dim // 8, kernel_size=2)
        self.layer_1_1 = nn.Sequential(self.conv1_1, self.bn1_1, self.conv1_2, self.bn1_2, self.conv1_3, self.conv1_4, self.global_avg_pool)

        self.conv2_1 = dynamic_conv2d(conv_in_dim, conv_in_dim // 2, kernel_size=3, padding=1, stride=2)
        self.conv2_2 = dynamic_conv2d(conv_in_dim // 2, conv_in_dim // 4, kernel_size=3, padding=1, stride=2)
        self.bn2_1 = nn.BatchNorm2d(conv_in_dim // 2)
        self.bn2_2 = nn.BatchNorm2d(conv_in_dim // 4)
        self.conv2_3 = dynamic_conv2d(conv_in_dim // 4, conv_in_dim // 4, kernel_size=2)
        self.layer_1_2 = nn.Sequential(self.conv2_1, self.bn2_1, self.conv2_2, self.bn2_2, self.conv2_3, self.global_avg_pool)
        
        self.conv3_1 = dynamic_conv2d(conv_in_dim, conv_in_dim // 4, kernel_size=3)
        self.bn3_1 = nn.BatchNorm2d(conv_in_dim // 4)
        self.conv3_2 = dynamic_conv2d(conv_in_dim // 4, conv_in_dim // 8, kernel_size=3)
        self.bn3_2 = nn.BatchNorm2d(conv_in_dim // 8)
        self.conv3_3 = dynamic_conv2d(conv_in_dim // 8, conv_in_dim // 16, kernel_size=3)
        self.layer_1_3 = nn.Sequential(self.conv3_1, self.bn3_1, self.conv3_2, self.bn3_2, self.conv3_3, self.global_avg_pool)
        
        self.weights_function = nn.Sequential(nn.Linear(224, 1), nn.Sigmoid())

    def forward(self, x):
        y1 = self.layer_1_1(x)
        y2 = self.layer_1_2(x)
        y3 = self.layer_1_3(x)
        z = torch.cat([y1, y2, y3], 1)
        z = self.faltten(z)
        attention_weights = self.weights_function(z)
        return attention_weights

class fc_net(nn.Module):
    def __init__(self, fc_in_features = 1, num_class = 7):
        super(fc_net, self).__init__()
        self.fc1 = nn.Linear(fc_in_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, num_class)
    
    def forward(self, x):
        return self.fc4(self.fc3(self.fc2(self.fc1(x))))

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net[2](net[1](net[0](X))).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def run_training():
    args = basic_parameters()
    imagenet_pretrained = True
    res18 = basic_net(imagenet_pretrained, args.drop_rate) 
    attention = attention_net()
    fc = fc_net()

    if args.pretrained:
        print("Loading pretrained weights...", args.pretrained) 
        pretrained = torch.load(args.pretrained)
        pretrained_state_dict = pretrained['state_dict']
        model_state_dict = res18.state_dict()
        loaded_keys = 0
        total_keys = 0
        for key in pretrained_state_dict:
            if  ((key=='module.fc.weight')|(key=='module.fc.bias')):
                pass
            else:    
                model_state_dict[key] = pretrained_state_dict[key]
                total_keys+=1
                if key in model_state_dict:
                    loaded_keys+=1
        print("Loaded params num:", loaded_keys)
        print("Total params num:", total_keys)
        res18.load_state_dict(model_state_dict, strict = False)  
        attention.load_state_dict(model_state_dict, strict=False)
        fc.load_state_dict(model_state_dict, strict=False)
        
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02,0.25))])
    
    train_dataset = raf_datasets(args.raf_path, phase = 'train', transform = data_transforms, basic_aug = True)    
    
    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               shuffle = True,  
                                               pin_memory = True)

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])                                           
    val_dataset = raf_datasets(args.raf_path, phase = 'test', transform = data_transforms_val)    
    print('Validation set size:', val_dataset.__len__())
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               shuffle = False,  
                                               pin_memory = True)
    
    params = [attention.parameters(), fc.parameters()]
    optimizer_attention = torch.optim.Adam(params[0], weight_decay = 1e-4)
    scheduler_attention = torch.optim.lr_scheduler.ExponentialLR(optimizer_attention, gamma = 0.9)
    optimizer_fc = torch.optim.Adam(params[1], weight_decay = 1e-4)
    scheduler_fc = torch.optim.lr_scheduler.ExponentialLR(optimizer_fc, gamma = 0.9)
    
    #res18 = res18.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    
    margin_1 = args.margin_1
    margin_2 = args.margin_2
    beta = args.beta
    
    for i in range(1, args.epochs + 1):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        res18.train()
        attention.train()
        fc.train()
        
        for batch_i, (imgs, targets, indexes) in enumerate(train_loader):
            batch_sz = imgs.size(0)
            iter_cnt += 1
            tops = int(batch_sz* beta)
            optimizer_attention.zero_grad()
            optimizer_fc.zero_grad()
            
            res18_out = res18(imgs)
            attention_out = attention(res18_out)
            fc_out = fc(attention_out)
            outputs = fc_out * attention_out
            
            _, top_idx = torch.topk(attention_out.squeeze(), tops)
            _, down_idx = torch.topk(attention_out.squeeze(), batch_sz - tops, largest = False)
            high_group = attention_out[top_idx]
            low_group = attention_out[down_idx]
            high_mean = torch.mean(high_group)
            low_mean = torch.mean(low_group)
            diff  = low_mean - high_mean + margin_1
            
            if diff > 0:
                RR_loss = diff
            else:
                RR_loss = 0.0
            
            loss = criterion(outputs, targets) + RR_loss 
            loss.backward()
            optimizer_attention.step()
            optimizer_fc.step()
            
            running_loss += loss
            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num
            
            if i >= args.relabel_epoch:
                sm = torch.softmax(outputs, dim = 1)
                Pmax, predicted_labels = torch.max(sm, 1)
                Pgt = torch.gather(sm, 1, targets.view(-1,1)).squeeze()
                true_or_false = Pmax - Pgt > margin_2
                update_idx = true_or_false.nonzero().squeeze()
                label_idx = indexes[update_idx]
                relabels = predicted_labels[update_idx]
                train_loader.dataset.label[label_idx.cpu().numpy()] = relabels.cpu().numpy()
            
        scheduler_attention.step()
        scheduler_fc.step()
        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt
        val_acc = evaluate_accuracy(val_loader, [res18, attention, fc])
        print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f  Val accuracy' % (i, acc, running_loss, val_acc))

if __name__ == "__main__":
    run_training()
