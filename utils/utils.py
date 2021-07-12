import torch.nn.functional as F
import time
import itertools
from torch.autograd import Variable
from tqdm import tqdm
import torch
import numpy as np

def compute_result_image(dataloader, net):
    bs, clses = [], []

    time_start = time.time()
    # for batch_idx, (img_names, images, labels) in enumerate(dataloader):
    for batch_idx, (images, labels) in enumerate(dataloader):
        clses.append(labels.data.cpu())
        with torch.no_grad():
            images, labels = Variable(images).cuda(), Variable(labels).cuda()

        hashCodes = net.forward(images)
        # hashCodes = net_2.forward(hashFeatures)
        hashCodes = torch.tanh(hashCodes)
        bs.append(hashCodes.data.cpu())

    total_time = time.time() - time_start

    return torch.sign(torch.cat(bs)), torch.cat(clses), total_time
    # return torch.cat(img_name), torch.cat(img),  torch.sign(torch.cat(bs)), torch.cat(clses), total_time


def triplet_loss(Ihash, labels, margin):#单模态的triplet loss
    triplet_loss = torch.tensor(0.0).cuda()
    labels_ = labels.cpu().data.numpy()
    triplets = []
    for label in labels_:
        label_mask = np.matmul(labels_, np.transpose(label)) > 0  # multi-labels
        label_indices = np.where(label_mask)[0]
        if len(label_indices) < 2:
            continue
        negative_indices = np.where(np.logical_not(label_mask))[0]
        if len(negative_indices) < 1:
            continue
        anchor_positives = list(itertools.combinations(label_indices, 2))
        temp = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                for neg_ind in negative_indices]
        triplets += temp

    length = len(triplets)

    if triplets:
        triplets = np.array(triplets)
        # print('triplet', triplets.shape)
        # intra triplet loss
        I_ap = (Ihash[triplets[:, 0]] - Ihash[triplets[:, 1]]).pow(2).sum(1)
        I_an = (Ihash[triplets[:, 0]] - Ihash[triplets[:, 2]]).pow(2).sum(1)
        # print('I_ap = ', I_ap)
        # print('I_an = ', I_an)
        triplet_loss = F.relu(margin + I_ap - I_an).mean()

    return triplet_loss, length


def CrossModel_triplet_loss(imgae_Ihash, text_Ihash, labels, margin):
    triplet_loss = torch.tensor(0.0).cuda()
    labels_ = labels.cpu().data.numpy()
    triplets = []
    for label in labels_:
        label_mask = np.matmul(labels_, np.transpose(label)) > 0  # multi-labels  (32,255)x(255,1) =（32,1） 某一行>0说明这两个图片的标签有重合，是同类样本
        label_indices = np.where(label_mask)[0]   #np.where 输出满足条件 (即非0) 元素的坐标  与当前样本属于同类的样本（包含它自己）
        if len(label_indices) < 2:
            continue
        negative_indices = np.where(np.logical_not(label_mask))[0]
        if len(negative_indices) < 1:
            continue
        anchor_positives = list(itertools.combinations(label_indices, 2))
        temp = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                for neg_ind in negative_indices]
        triplets += temp

    length = len(triplets)

    if triplets:
        triplets = np.array(triplets)
        # print('triplet', triplets.shape)
        # intra triplet loss
        # image
        imgae_I_ap = (imgae_Ihash[triplets[:, 0]] - imgae_Ihash[triplets[:, 1]]).pow(2).sum(1)
        imgae_I_an = (imgae_Ihash[triplets[:, 0]] - imgae_Ihash[triplets[:, 2]]).pow(2).sum(1)
        imgae_triplet_loss = F.relu(margin + imgae_I_ap - imgae_I_an).mean()
        # text
        text_I_ap = (text_Ihash[triplets[:, 0]] - text_Ihash[triplets[:, 1]]).pow(2).sum(1)
        text_I_an = (text_Ihash[triplets[:, 0]] - text_Ihash[triplets[:, 2]]).pow(2).sum(1)
        text_triplet_loss = F.relu(margin + text_I_ap - text_I_an).mean()

        # cross model triplet loss
        # anchor-image    negative,positive-text
        imgae_text_I_ap = (imgae_Ihash[triplets[:, 0]] - text_Ihash[triplets[:, 1]]).pow(2).sum(1)
        imgae_text_I_an = (imgae_Ihash[triplets[:, 0]] - text_Ihash[triplets[:, 2]]).pow(2).sum(1)
        imgae_text_triplet_loss = F.relu(margin + imgae_text_I_ap - imgae_text_I_an).mean()
        # anchor-text     negative,positive-image
        text_image_I_ap = (text_Ihash[triplets[:, 0]] - imgae_Ihash[triplets[:, 1]]).pow(2).sum(1)
        text_image_I_an = (text_Ihash[triplets[:, 0]] - imgae_Ihash[triplets[:, 2]]).pow(2).sum(1)
        text_image_triplet_loss = F.relu(margin + text_image_I_ap - text_image_I_an).mean()

    return imgae_triplet_loss, text_triplet_loss, imgae_text_triplet_loss, text_image_triplet_loss, length


def compute_result_CrossModel(dataloader, text_net, image_net):
    bs_image, bs_text, clses = [], [], []

    time_start = time.time()

    for batch_idx, (images, texts, labels) in tqdm(enumerate(dataloader)):
        clses.append(labels.data.cpu())

        # image
        with torch.no_grad():
            images, texts, labels = Variable(images).cuda(), Variable(texts).cuda(),Variable(labels).cuda()
        image_hashCodes = image_net.forward(images)
        # image_hashCodes = torch.tanh(image_hashCodes)


        text_hashCodes = text_net.forward(texts)

        bs_image.append(image_hashCodes.data.cpu())
        bs_text.append(text_hashCodes.data.cpu())

    total_time = time.time() - time_start

    return torch.sign(torch.cat(bs_image)), torch.sign(torch.cat(bs_text)), torch.cat(clses), total_time

def compute_mAP_MultiLabels(trn_binary, tst_binary, trn_label, tst_label):
    """
    compute mAP by searching testset from trainset
    https://github.com/flyingpot/pytorch_deephash
    """
    for x in trn_binary, tst_binary, trn_label, tst_label: x.long()

    AP = []
    Ns = torch.arange(1, trn_binary.size(0) + 1)
    Ns = Ns.type(torch.FloatTensor)
    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        _, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort() #计算汉明距离，并将距离从小到大排序(query_result是索引)
        # correct = (query_label == trn_label[query_result]).float()
        correct = ((trn_label[query_result] * query_label).sum(1) > 0).float() #这18000个索引对应的样本label是否与query label有重合
        P = torch.cumsum(correct.type(torch.FloatTensor), dim=0) / Ns
        AP.append(torch.sum(P * correct) / torch.sum(correct))
    mAP = torch.mean(torch.Tensor(AP))
    return mAP

# def compute_mAP_MultiLabels(trn_binary, tst_binary, trn_label, tst_label):
#     """
#     compute mAP by searching testset from trainset
#     https://github.com/flyingpot/pytorch_deephash
#     """
#     for x in trn_binary, tst_binary, trn_label, tst_label: x.long()
#
#     AP = []
#     Ns = torch.arange(1, trn_binary.size(0) + 1)
#     Ns = Ns.type(torch.FloatTensor)
#     # cnt = 0
#     # total = 0.0
#     # print('Ns = ', Ns)
#     for i in range(tst_binary.size(0)):
#         query_label, query_binary = tst_label[i], tst_binary[i]
#         # print('query_binary = ', query_binary)
#         # print('trn_binary = ', trn_binary)
#         # 计算汉明距离，并将距离从小到大排序(query_result是索引)
#         _, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort()
#         # print('query_result = ', query_result)
#         # correct = (query_label == trn_label[query_result]).float()
#         # 与 query label 相同的
#         correct = ((trn_label[query_result] * query_label).sum(1) > 0).float()
#         # print('correct = ', correct)
#         P = torch.cumsum(correct, dim=0) / Ns
#         # print('P = ', P)
#         AP.append(torch.sum(P * correct) / torch.sum(correct))
#
#         # if query_label[0] == 1:
#         #     print('tst'+str(i)+' AP = ', torch.sum(P * correct) / torch.sum(correct))
#         # cnt += 1
#         # total += torch.sum(P * correct) / torch.sum(correct)
#         # print('append to AP = ', torch.sum(P * correct) / torch.sum(correct))
#         # print(torch.sum(correct))
#         # print(trn_binary.size(0))
#     # print('AP = ', AP)
#     mAP = torch.mean(torch.Tensor(AP))
#     # print('cnt = ', cnt)
#     # print('total = ', total)
#     # print('mAP = ', mAP)
#     return mAP

