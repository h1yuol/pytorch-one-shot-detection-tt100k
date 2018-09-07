import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

from lib.model.one_shot.one_shot_resnet import one_shot_resnet
from lib.datasets.data_loader import get_dataloaders

if __name__ == '__main__':
    model_path = '/home/huangyucheng/MYDATA/EXPERIMENTS/FasterRCNN/2gpu_0831/res101/tt100k/faster_rcnn_1_50_2032.pth'

    margin = 0.1
    model = one_shot_resnet(128, model_path, margin, pretrained=True)

    model.create_architecture()
    # model.test()

    # get dataloaders
    num_workers = {
        'train': 9,
        'test': 6,
    }
    P, K = 32, 4
    dataloaders, dataset_sizes = get_dataloaders(num_workers, P, K, ['train','test'])

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lr = 0.001
    params = []
    for key,value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params.append({
                        'params': [value],
                        'lr': lr*2,
                        'weight_decay': 0,
                    })
            else :
                params.append({
                        'params': [value],
                        'lr': lr,
                        'weight_decay': 5e-4,
                    })
    optimizer = optim.SGD(params, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model.cuda()
    im_data = torch.FloatTensor(1).cuda()
    labels = torch.FloatTensor(1).cuda()
    im_data = Variable(im_data)
    labels = Variable(labels)

    num_epochs = 100
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-'*10)

        for phase in ['train','test']:
            if phase == 'train':
                exp_lr_scheduler.step()
                model.train()
            else: 
                model.eval()

            running_loss = 0.0

            for data in dataloaders[phase]:
                im_data.data.resize_(data[0].size()).copy_(data[0])
                labels.data.resize_(data[1].size()).copy_(data[1])

                optimizer.zero_grad()

                # forward
                embeddings, loss = model(im_data,labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * im_data.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss {:.4f}'.format(phase, epoch_loss))


    from IPython import embed
    embed()









