from PIL import Image, ImageFont, ImageDraw
import numpy as np
import torchvision.models as models
import torch.utils.data as data
from torchvision import transforms
import cv2
import pandas as pd
import os
import torch
import torch.nn as nn
import random
import deal_with_photo
import torch.nn.functional as F



class Basic_Parameters:
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
        self.facial_label = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
        if self.pretrained:
            self.pretrained_path = './model.pkl'
        else:
            self.pretrained_path = None


class Raf_Datasets(data.Dataset):
    def __init__(self, raf_path, phase, transform=None, basic_aug=False):
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
            f = f + "_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image', f)
            self.file_paths.append(path)

        self.basic_aug = basic_aug
        self.aug_func = [deal_with_photo.flip_image, deal_with_photo.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]
        label = self.label[idx]
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx


class Global_Avgpool2d(nn.Module):
    def __init__(self):
        super(Global_Avgpool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class Flatten_Layer(nn.Module):
    def __init__(self):
        super(Flatten_Layer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Attention2d(nn.Module):
    def __init__(self, in_planes, k):
        super(Attention2d, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, k, 1)
        self.fc2 = nn.Conv2d(k, k, 1)

    def forward(self, x):
        print(x.size())
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.view(x.size(0), -1)
        x = F.softmax(x)
        return x


class Dynamic_Conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4):
        super(Dynamic_Conv2d, self).__init__()
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
        self.attention = Attention2d(in_planes, K)

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


class Residual_Block (nn.Module):
    def __init__(self, i_channel, o_channel, kernel_size=3, stride=1):
        super(Residual_Block, self).__init__()
        self.i = i_channel
        self.o = o_channel
        self.k = kernel_size
        if (self.i != self.o) or stride != 1:
            self.change_channel = nn.Sequential(
                Dynamic_Conv2d(self.i, self.o, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.o)
            )
        else:
            self.change_channel = None

        self.conv1 = Dynamic_Conv2d(i_channel, o_channel, kernel_size=self.k, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(o_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = Dynamic_Conv2d(o_channel, o_channel, kernel_size=self.k, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(o_channel)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.change_channel:
            residual = self.change_channel(residual)

        out += residual
        out = self.relu(out)
        return out


class Basic_Net(nn.Module):
    def __init__(self, pretrained=True, drop_rate=0):
        super(Basic_Net, self).__init__()
        self.drop_rate = drop_rate
        resnet = models.resnet18(pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc_in_dim = list(resnet.children())[-1].in_features
        self.features = self.features[: -1]

    def forward(self, x):
        x = self.features(x)
        if self.drop_rate > 0:
            x = nn.Dropout(self.drop_rate)(x)
        return x


class Attention_Net(nn.Module):
    def __init__(self, conv_in_dim=512):
        super(Attention_Net, self).__init__()
        self.d = conv_in_dim
        self.faltten = Flatten_Layer()
        self.global_avg_pool = Global_Avgpool2d()

        self.conv1_1 = Residual_Block(self.d, self.d // 2)
        self.conv1_2 = Residual_Block(self.d // 2, self.d // 4)
        self.conv1_3 = Residual_Block(self.d // 4, self.d // 4)
        self.conv1_4 = Residual_Block(self.d // 4, self.d // 8)
        self.bn1_1 = nn.BatchNorm2d(self.d // 2)
        self.bn1_2 = nn.BatchNorm2d(self.d // 4)
        self.layer1 = nn.Sequential(self.conv1_1, self.bn1_1, self.conv1_2, self.bn1_2,
                                    self.conv1_3, self.conv1_4, self.global_avg_pool)

        self.conv2_1 = Residual_Block(self.d, self.d // 2)
        self.conv2_2 = Residual_Block(self.d // 2, self.d // 4)
        self.conv2_3 = Residual_Block(self.d // 4, self.d // 4)
        self.bn2_1 = nn.BatchNorm2d(self.d // 2)
        self.bn2_2 = nn.BatchNorm2d(self.d // 4)
        self.layer2 = nn.Sequential(self.conv2_1, self.bn2_1, self.conv2_2,
                                    self.bn2_2, self.conv2_3, self.global_avg_pool)

        self.conv3_1 = Residual_Block(self.d, self.d // 4)
        self.conv3_2 = Residual_Block(self.d // 4, self.d // 8)
        self.conv3_3 = Residual_Block(self.d // 8, self.d // 16)
        self.bn3_1 = nn.BatchNorm2d(self.d // 4)
        self.bn3_2 = nn.BatchNorm2d(self.d // 8)
        self.layer3 = nn.Sequential(self.conv3_1, self.bn3_1, self.conv3_2,
                                    self.bn3_2, self.conv3_3, self.global_avg_pool)
        self.weights_function = nn.Sequential(nn.Linear(224, 1), nn.Sigmoid())

    def forward(self, x):
        y1 = self.layer1(x)
        y2 = self.layer2(x)
        y3 = self.layer3(x)
        z = torch.cat([y1, y2, y3], 1)
        z = self.faltten(z)
        attention_weights = self.weights_function(z)
        return attention_weights


class FC_Net(nn.Module):
    def __init__(self, fc_in_features=1, num_class=7):
        super(FC_Net, self).__init__()
        self.fc1 = nn.Linear(fc_in_features, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 7)
        self.fc4 = nn.Linear(7, num_class)

    def forward(self, x):
        return self.fc4(self.fc3(self.fc2(self.fc1(x))))


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        feature = net[0](X)
        attention = net[1](feature)
        classify = net[2](attention)
        result = attention * classify
        acc_sum += (result.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def get_transform(tag=0):
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25))])
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    if tag == 0:
        return data_transforms
    else:
        return data_transforms_val
    

def run_training():
    imagenet_pretrained = True
    res18 = Basic_Net(imagenet_pretrained, args.drop_rate)
    attention = Attention_Net()
    fc = FC_Net()

    if args.pretrained:
        print("Loading pretrained weights...", args.pretrained)
        pretrained = torch.load(args.pretrained_path)
        res18_state_dict = pretrained['res18']
        atten_state_dict = pretrained['atten']
        fc_state_dict = pretrained['fc']

        res18.load_state_dict(res18_state_dict, strict=False)
        print('res18_net load success...')
        attention.load_state_dict(atten_state_dict, strict=False)
        print('Attention_Net load success...')
        fc.load_state_dict(fc_state_dict, strict=False)
        print('FC_Net load success...')

        res18.eval()
        attention.eval()
        fc.eval()
        return res18, attention, fc

    data_transforms = get_transform()

    train_dataset = Raf_Datasets(args.raf_path, phase='train', transform=data_transforms, basic_aug=True)

    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True)

    data_transforms_val = get_transform(1)
    val_dataset = Raf_Datasets(args.raf_path, phase='test', transform=data_transforms_val)
    print('Validation set size:', val_dataset.__len__())

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True)

    params = [attention.parameters(), fc.parameters()]
    optimizer_attention = torch.optim.Adam(params[0], weight_decay=1e-4)
    scheduler_attention = torch.optim.lr_scheduler.ExponentialLR(optimizer_attention, gamma=0.9)
    optimizer_fc = torch.optim.Adam(params[1], weight_decay=1e-4)
    scheduler_fc = torch.optim.lr_scheduler.ExponentialLR(optimizer_fc, gamma=0.9)

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
            tops = int(batch_sz * beta)
            optimizer_attention.zero_grad()
            optimizer_fc.zero_grad()

            res18_out = res18(imgs)
            attention_out = attention(res18_out)
            fc_out = fc(attention_out)
            outputs = fc_out * attention_out

            _, top_idx = torch.topk(attention_out.squeeze(), tops)
            _, down_idx = torch.topk(attention_out.squeeze(), batch_sz - tops, largest=False)
            high_group = attention_out[top_idx]
            low_group = attention_out[down_idx]
            high_mean = torch.mean(high_group)
            low_mean = torch.mean(low_group)
            diff = low_mean - high_mean + margin_1

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
                sm = torch.softmax(outputs, dim=1)
                p_max, predicted_labels = torch.max(sm, 1)
                p_gt = torch.gather(sm, 1, targets.view(-1, 1)).squeeze()
                true_or_false = p_max - p_gt > margin_2
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
        save_model(res18, attention, fc, args.pretrained_path)
    return res18, attention, fc

def save_model(res18, attention, fc, path):
    save_dict = {
        'res18': res18.state_dict(),
        'attention': attention.state_dict(),
        'fc': fc.state_dict()
    }
    torch.save(
        save_dict,
        path
    )


def catch_face(frame, net=None):
    classfier = cv2.CascadeClassifier('d:/Anacoda/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
    color = (0, 255, 0)
    grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    face_rects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(face_rects) > 0:
        for face_rect in face_rects:
            x, y, w, h = face_rect
            if x < 10:
                x = 10
            if y < 10:
                y = 10
            image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
            pil_img = cv2pil(image)
            label = predict_model(pil_img, net)
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
            frame = paint(frame, args.facial_label[label], (x - 10, y + h + 10), color)
    return frame


def cv2pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def predict_model(image, net=None):
    data_transform = get_transform(1)
    image = data_transform(image)
    image = image.view(-1, 3, 224, 224)
    out = net[0](image)
    atten_out = net[1](out)
    fc_out = net[2](atten_out)
    out = fc_out * atten_out
    pred = out.max(1, keepdim=True)[1]
    return pred.item()


def paint(im, chinese, pos, color):
    img_pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('c:/Windows/Fonts/STZHONGS.TTF', 20)
    fill_color = color
    position = pos
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, chinese, font=font, fill=fill_color)
    img = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
    return img


def recognize_video(window_name='face recognize', camera_idx=0, net=None):
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(camera_idx)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        catch_frame = catch_face(frame, net)
        cv2.imshow(window_name, catch_frame)
        if cv2.waitKey(100) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = Basic_Parameters()
    res18, attention, fc = run_training()
    recognize_video(net=[res18, attention, fc])
