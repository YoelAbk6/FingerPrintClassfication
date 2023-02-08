# import torch
# import torchvision.models as models
# import torch.nn as nn

# class DDCresnet50(nn.Module):
#     def _init_(self, num_classes=2, pretrained=True):
#         super(DDCresnet50, self)._init_()

#         # Load the pre-trained ResNet50 model
#         base_model = models.resnet50(pretrained=pretrained)
        
#         # Replace the standard convolutions with dense convolutions
#         dense_conv = nn.Conv2d(in_channels=base_model.conv1.in_channels,
#                                out_channels=base_model.conv1.out_channels,
#                                kernel_size=base_model.conv1.kernel_size,
#                                stride=base_model.conv1.stride,
#                                padding=base_model.conv1.padding,
#                                bias=False)
#         dense_conv.weight.data = base_model.conv1.weight.data
#         self.conv1 = dense_conv
        
#         # Replace the remaining convolutions with dense convolutions
#         self.layer1 = base_model.layer1
#         self.layer2 = base_model.layer2
#         self.layer3 = base_model.layer3
#         self.layer4 = base_model.layer4
#         self.avgpool = base_model.avgpool
        
#         # Add a fully connected layer for the final classification
#         self.fc = nn.Linear(in_features=base_model.fc.in_features,
#                             out_features=num_classes)
        
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x

# # Create the dense ResNet50 model
# model = DDCresnet50(num_classes=2)

# # Move the model to GPU
# model = model.to(device)

# # Define the loss function, optimizer, and training loop
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# # Train the model on your fingerprint data
# for epoch in range(100):
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data
#         inputs = inputs.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()

#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f