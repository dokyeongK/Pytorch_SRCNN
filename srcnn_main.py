import argparse
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np
import srcnn_data_loader as SRCNN_DATA
import torch.utils.data as data
import srcnn_model
import scipy.misc
from math import log10

def saveImg(img, outFileName):
    img = img / 2 + 0.5
    npimg = img.numpy()
    scipy.misc.imsave(outFileName, np.transpose(npimg, (1, 2, 0)))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch SRCNN')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--upscale', type=int, default=2, help='Scale factor')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args()

    print(args)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = SRCNN_DATA.SRCNN(root_dir='./dataset/train', transform=transform, upscale=args.upscale)

    trainloader = data.DataLoader(trainset,
                                  shuffle=True,
                                  num_workers=1)

    testset = SRCNN_DATA.SRCNN(root_dir='./dataset/test', transform=transform, upscale=args.upscale)

    testloader = data.DataLoader(testset,
                                  shuffle=True,
                                  num_workers=1)

    model = srcnn_model.SRCNN()

    loss_func = nn.MSELoss()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model.cuda()
    loss_func = loss_func.cuda()
    optimizer = opt.Adam(model.parameters(), lr=args.lr)

    for e in range(args.epochs):
     epoch_loss = 0
     for itr, data in enumerate(trainloader):
        imgLR, imgHR = data

        if use_cuda:
            imgLR = imgLR.cuda()
            imgHR = imgLR.cuda()

        optimizer.zero_grad()
        out_model = model(imgLR)
        loss = loss_func(out_model, imgHR)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
     print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(e, epoch_loss / len(trainset)))

#       Test part

    sum_psnr = 0
    for itr, data in enumerate(testloader):
        imgLR, imgHR = data

        if use_cuda:
            imgLR = imgLR.cuda()
            imgHR = imgLR.cuda()

        sr_result = model(imgLR)

        if use_cuda:
            outImg = sr_result.data.cpu().squeeze(0)
        else:
            outImg = sr_result.data.squeeze(0)

        outFileName = 'output/out_' + 'epoch_' + str(args.epochs) + '_' + str(itr) + '.jpg'
        saveImg(outImg, outFileName)

        MSE = loss_func(sr_result, imgHR)
        psnr = 10 * log10(1 / MSE.item())
        sum_psnr += psnr

    print("**Average PSNR: {} dB".format(sum_psnr / len(testloader)))

