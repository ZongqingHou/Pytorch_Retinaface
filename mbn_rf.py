import torch
from torch import nn
import torch.nn.functional as F


in_channels_stage2 = 32
in_channels_list = [
	in_channels_stage2 * 2,
	in_channels_stage2 * 4,
	in_channels_stage2 * 8,
]
out_channels = 64

class SSH(nn.Module):
	def __init__(self, in_channel, out_channel):
		super(SSH, self).__init__()
		self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

		self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1)
		self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

		self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1)
		self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

	def forward(self, input):
		conv3X3 = self.conv3X3(input)

		conv5X5_1 = self.conv5X5_1(input)
		conv5X5 = self.conv5X5_2(conv5X5_1)

		conv7X7_2 = self.conv7X7_2(conv5X5_1)
		conv7X7 = self.conv7x7_3(conv7X7_2)

		out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
		out = F.relu(out)
		return out


class ClassHead(nn.Module):
	def __init__(self,inchannels=512,num_anchors=3):
		super(ClassHead,self).__init__()
		self.num_anchors = num_anchors
		self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

	def forward(self,x):
		out = self.conv1x1(x)
		# out_ = out.permute(0,2,3,1).contiguous()
		# out_ = out_.view(out_.shape[0], -1, 2)
		
		return out

class BboxHead(nn.Module):
	def __init__(self,inchannels=512,num_anchors=3):
		super(BboxHead,self).__init__()
		self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

	def forward(self,x):
		out = self.conv1x1(x)
		# out = out.permute(0,2,3,1).contiguous()

		# return out.view(out.shape[0], -1, 4)
		return out

class LandmarkHead(nn.Module):
	def __init__(self,inchannels=512,num_anchors=3):
		super(LandmarkHead,self).__init__()
		self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

	def forward(self,x):
		out = self.conv1x1(x)
		# out = out.permute(0,2,3,1).contiguous()

		# return out.view(out.shape[0], -1, 10)
		return out


def conv_dw(inp, oup, stride):
	return nn.Sequential(
		nn.Conv2d(in_channels=inp, out_channels=inp, kernel_size=3, stride=stride, padding=1, groups=inp, bias=False),
		nn.BatchNorm2d(inp),
		nn.LeakyReLU(0.1),
		nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, stride=1, padding=0, bias=False),
		nn.BatchNorm2d(oup),
		nn.LeakyReLU(0.1),
	)


def conv_bn1X1(inp, oup, stride):
	return nn.Sequential(
		nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
		nn.BatchNorm2d(oup),
		nn.LeakyReLU(0.1)
	)


def conv_bn(inp, oup, stride=1):
	return nn.Sequential(
		nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=3, stride=stride, padding=1, bias=False),
		nn.BatchNorm2d(oup),
		nn.LeakyReLU(0.1)
	)


def conv_bn_no_relu(inp, oup, stride):
	return nn.Sequential(
		nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=3, stride=stride, padding=1, bias=False),
		nn.BatchNorm2d(oup),
	)


class Mbn_RF(nn.Module):
	def __init__(self):
		super(Mbn_RF, self).__init__()
		# MobileNet Part
		self.stage1 = nn.Sequential(
			conv_bn(3, 8, 2),    # 3
			conv_dw(8, 16, 1),   # 7
			conv_dw(16, 32, 2),  # 11
			conv_dw(32, 32, 1),  # 19
			conv_dw(32, 64, 2),  # 27
			conv_dw(64, 64, 1),  # 43
		)
		self.stage2 = nn.Sequential(
			conv_dw(64, 128, 2),  # 43 + 16 = 59
			conv_dw(128, 128, 1), # 59 + 32 = 91
			conv_dw(128, 128, 1), # 91 + 32 = 123
			conv_dw(128, 128, 1), # 123 + 32 = 155
			conv_dw(128, 128, 1), # 155 + 32 = 187
			conv_dw(128, 128, 1), # 187 + 32 = 219
		)
		self.stage3 = nn.Sequential(
			conv_dw(128, 256, 2), # 219 +3 2 = 241
			conv_dw(256, 256, 1), # 241 + 64 = 301
		)

		# FPN Part
		self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride=1)
		self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride=1)
		self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride=1)

		self.merge1 = conv_bn(out_channels, out_channels)
		self.merge2 = conv_bn(out_channels, out_channels)

		# SSH Part
		self.ssh1 = SSH(out_channels, out_channels)
		self.ssh2 = SSH(out_channels, out_channels)
		self.ssh3 = SSH(out_channels, out_channels)

		# Header Part
		self.classhead_1 = ClassHead(64, 2)
		self.classhead_2 = ClassHead(64, 2)
		self.classhead_3 = ClassHead(64, 2)

		self.bboxhead_1 = BboxHead(64, 2)
		self.bboxhead_2 = BboxHead(64, 2)
		self.bboxhead_3 = BboxHead(64, 2)

		self.ldmhead_1 = LandmarkHead(64, 2)
		self.ldmhead_2 = LandmarkHead(64, 2)
		self.ldmhead_3 = LandmarkHead(64, 2)

	def forward(self, x):
		x_1 = self.stage1(x)
		x_2 = self.stage2(x_1)
		x_3 = self.stage3(x_2)

		output1 = self.output1(x_1)
		output2 = self.output2(x_2)
		output3 = self.output3(x_3)

		up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
		output2 = output2 + up3
		output2 = self.merge2(output2)

		up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
		output1 = output1 + up2
		output1 = self.merge1(output1)

		output1 = self.ssh1(output1)
		output2 = self.ssh2(output2)
		output3 = self.ssh3(output3)

		class1, bbox1, ldm1 = self.classhead_1(output1), self.bboxhead_1(output1), self.ldmhead_1(output1)
		class2, bbox2, ldm2 = self.classhead_2(output2), self.bboxhead_2(output2), self.ldmhead_2(output2)
		class3, bbox3, ldm3 = self.classhead_3(output3), self.bboxhead_3(output3), self.ldmhead_3(output3)

		return class1, bbox1, ldm1, class2, bbox2, ldm2, class3, bbox3, ldm3


if __name__ == "__main__":
	model = Mbn_RF()
	model_dict = {}
	net_dict = torch.load("./weights/mobilenet0.25_Final.pth")

	for k, v in net_dict.items():
		if "body." in k:
			model_dict[k.split("body.")[-1]] = v
		elif "fpn." in k:
			model_dict[k.split("fpn.")[-1]] = v
		elif "BboxHead.0." in k:
			model_dict[k.replace("BboxHead.0", "bboxhead_1")] = v
		elif "BboxHead.1." in k:
			model_dict[k.replace("BboxHead.1", "bboxhead_2")] = v
		elif "BboxHead.2." in k:
			model_dict[k.replace("BboxHead.2", "bboxhead_3")] = v
		elif "LandmarkHead.0." in k:
			model_dict[k.replace("LandmarkHead.0", "ldmhead_1")] = v
		elif "LandmarkHead.1." in k:
			model_dict[k.replace("LandmarkHead.1", "ldmhead_2")] = v
		elif "LandmarkHead.2." in k:
			model_dict[k.replace("LandmarkHead.2", "ldmhead_3")] = v
		elif "ClassHead.0." in k:
			model_dict[k.replace("ClassHead.0", "classhead_1")] = v
		elif "ClassHead.1." in k:
			model_dict[k.replace("ClassHead.1", "classhead_2")] = v
		elif "ClassHead.2." in k:
			model_dict[k.replace("ClassHead.2", "classhead_3")] = v
		else:
			model_dict[k] = v
			print(k)
			print("-------------------")

	model.load_state_dict(model_dict)