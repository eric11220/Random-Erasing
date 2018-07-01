import argparse
import torch, os
import numpy as np

from torch import optim
from torch.autograd import Variable
from torchvision.transforms import transforms
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

import models.cifar as models
from models.cifar.resnet import ResnetConv

batch_size = 10
n_way = 20
k_shot = 5

data_dir = "five_shot"
train_csv = os.path.join(data_dir, "train.csv")
val_csv = os.path.join(data_dir, "val.csv")
img_dir = os.path.join(data_dir, "images")
test_img_dir = "../../data/test"

novel_start = 80
novel_class = np.arange(80, 100)
novel_mapping = ["00", "10", "23", "30", "32", \
                 "35", "48", "54", "57", "59", \
                 "60", "64", "66", "69", "71", \
                 "82", "91", "92", "93", "95"]

def parse_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', help="Saved checkpoint")
    parser.add_argument('--mode', help="Evaluate or testing", default="eval")
    parser.add_argument('--gather', help="Perform mean on support features", action="store_true")
    parser.add_argument('--n-neighbor', help="Number of KNN neighbors", default=5)
    return parser.parse_args()

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid
    from tensorboardX import SummaryWriter
    from datetime import datetime

    args = parse_inputs()

    # Multi-GPU support
    model = models.__dict__['resnet'](num_classes=80, depth=20)
    model = torch.nn.DataParallel(model).cuda()

    if not os.path.exists(args.ckpt):
        print("checkpoint doesnot exist... exiting...")
        exit

    print('load checkpoint ...', args.ckpt)
    checkpoint = torch.load(args.ckpt)
    model.load_state_dict(checkpoint['state_dict'])
    model = ResnetConv(model.module)
    model.eval()

    transform = transforms.Compose([ lambda x: Image.open(x).convert('RGB'),
                                               transforms.ToTensor() ])

    # Need to load all novel support images first
    support_cnt = {}
    support_x, support_y = [], []
    with open(train_csv, "r") as inf:
        for line in inf:
            path, cls = line.strip().split(",")
            path = os.path.join(img_dir, path)

            cls = int(cls)
            if cls not in novel_class:
                continue

            if support_cnt.get(cls, 0) >= k_shot:
                continue
            support_cnt[cls] = 1 if support_cnt.get(cls, 0) == 0 \
                                 else support_cnt[cls] + 1

            support_x.append(path)
            support_y.append(cls)

    support_x = np.asarray(support_x)
    support_y = np.asarray(support_y, dtype=np.uint8)

    support_x = [transform(path) for path in support_x]
    support_x = torch.stack(support_x)
    support_x = Variable(support_x).cuda()

    if args.mode == "eval":
        paths, labels = [], []
        with open(val_csv, 'r') as inf:
            for line in inf:
                name, cls = line.strip().split(',')
                cls = int(cls)

                paths.append(os.path.join(img_dir, name))
                labels.append(cls)
        labels = np.asarray(labels, dtype=np.uint8)
    else:
        img_ids = np.asarray([path.split('.')[0] for path in sorted(os.listdir(test_img_dir))], dtype=np.uint32)
        paths = [os.path.join(test_img_dir, path) for path in sorted(os.listdir(test_img_dir))]

    support_feats = model(support_x)
    support_feats = support_feats.view(support_feats.size(0), -1)
    support_feats = support_feats.data.cpu().numpy()

    if args.gather:
        centroids = []
        for cls in novel_class:
            ind = np.where(support_y == cls)
            feats = support_feats[ind]
            centroid = np.mean(feats, axis=0)
            centroids.append(centroid)
        centroids = np.asarray(centroids)

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(centroids, novel_class)
    else:
        centroids = support_feats
        knn = KNeighborsClassifier(n_neighbors=args.n_neighbor)
        knn.fit(centroids, support_y)

    choices = None
    start, total_correct = 0, 0
    while start < len(paths):
        end = start + batch_size

        query_x = torch.stack([transform(path) for path in paths[start:end]])
        query_x = query_x.cuda()

        query_feats = model(query_x)
        query_feats = query_feats.view(query_feats.size(0), -1)
        query_feats = query_feats.data.cpu().numpy()

        choice = knn.predict(query_feats)
        choices = choice if choices is None else np.concatenate((choices, choice))
        start += batch_size

    if args.mode == "eval":
        correct = np.sum(labels == choices)
        accu = correct / len(labels)
        print('<<<<>>>>accuracy:', accu)
    else:
        choices -= novel_start
        ind = np.argsort(img_ids)
        img_ids = img_ids[ind]
        choices = choices[ind]

        with open("pred.csv", "w") as outf:
            outf.write("image_id,predicted_label\n")
            for img_id, choice in zip(img_ids, choices):
                outf.write("%s,%s\n" % (img_id, novel_mapping[choice]))

