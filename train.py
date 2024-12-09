import os
import torch
import PointNets_model as pt
import data_processing as ds


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)




def pointNetLoss(ouputs, labels, m3x3, m64x64, alpha=0.0001):
    criterion = torch.nn.NLLLoss()
    bs =  ouputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)
    if ouputs.is_cuda:
        id3x3 = id3x3.cuda()
        id64x64 = id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3, m3x3.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64, m64x64.transpose(1, 2))
    return criterion(ouputs, labels) + alpha * (torch.norm(diff3x3) + torch.norm(diff64x64))/float(bs)


def train(pointnet, optimizer, train_loader, val_loader=None, epochs=15, save=True):
    best_val_acc = -1.0
    print(len(train_loader))
    for epoch in range(epochs):
        pointnet.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device).float()
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs, m3x3, m64x64 = pointnet(inputs.transpose(1, 2))
            loss = pointNetLoss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 50 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0


        pointnet.eval()
        correct = total = 0

        # validation
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs = inputs.to(device).float()
                labels = labels.to(device)
                outputs, __, __ = pointnet(inputs.transpose(1, 2))
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0) * labels.size(1)
                correct += (predicted == labels).sum().item()

        print("correct", correct, "/", total)
        val_acc = 100.0 * correct / total
        print('Valid accuracy: %d %%' % val_acc)

        # save the model
        if save and val_acc > best_val_acc:
            best_val_acc = val_acc
            path = os.path.join('', "pointnetmodel.yml")
            print("best_val_acc:", val_acc, "saving model at", path)
            torch.save(pointnet.state_dict(), path)


if __name__ == '__main__':
    pointnet = pt.PointNetSeg(classes=26)
    pointnet.to(device)
    dataset_path = '/storage/gtrivigno/geospatial/sem_kitti/dataset'
    optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.005)
    N_POINTS = 2000
    train_ds  = ds.PointCloudData(dataset_path, sequences=[1, 4, 7], num_points=N_POINTS)
    val_ds    = ds.PointCloudData(dataset_path, sequences=[3], num_points=N_POINTS)
    train_loader  = ds.DataLoader( dataset=train_ds,  batch_size=8, shuffle=True, drop_last=True)
    val_loader    = ds.DataLoader( dataset=val_ds,    batch_size=5, shuffle=False )
    train(pointnet, optimizer, train_loader, val_loader, save=True)
