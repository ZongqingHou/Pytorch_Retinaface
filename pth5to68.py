import torch
import torch.nn

pthfile = r'./weights/mobilenet0.25_nogroup_5.pth'
net_dict = torch.load(pthfile, map_location='cpu')
print(type(net_dict))

#resnet50+module.
for i in net_dict:

    # print(i)
    # print(net_dict[i].data.size())

    if i == 'module.LandmarkHead.0.conv1x1.weight':
        net_dict[i] = torch.randn(20*4, 64, 1, 1)
    if i == 'module.LandmarkHead.1.conv1x1.weight':
        net_dict[i] = torch.randn(20*4, 64, 1, 1)
    if i == 'module.LandmarkHead.2.conv1x1.weight':
        net_dict[i] = torch.randn(20*4, 64, 1, 1)
    if i == 'module.LandmarkHead.0.conv1x1.bias':
        net_dict[i] = torch.randn(20*4)
    if i == 'module.LandmarkHead.1.conv1x1.bias':
        net_dict[i] = torch.randn(20*4)
    if i == 'module.LandmarkHead.2.conv1x1.bias':
        net_dict[i] = torch.randn(20*4)


    print(i)
    print(net_dict[i].data.size())

torch.save(net_dict, './weights/mobilenet0.25_nogroup_20.pth')

