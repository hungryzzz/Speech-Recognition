from train_module import Options, Trainer

opt = Options().parse()
trainer = Trainer(opt)
trainer.train()
trainer.get_accuracy()


# from dataset import MyDataset

# from model import vgg11_bn, Mymodel
# from torch.utils.data import DataLoader

# dataset_save_path = 'dataset.pkl'
# dataset_path = "../dataset"

# dataset = MyDataset(dataset_path)
# print(len(dataset))

# print(dataset[0])

# class config(object):
#   input_s = dataset[0][0].shape[-1]
#   hidden_s = 256
#   max_len = dataset[0][0].shape[0]
#   dropout_rate = 0.5
#   num_class = 20
# model = Mymodel(config)

# output = model(dataset[0][0].unsqueeze(0))
# print(output.shape)
# print(output)


# for i in range(len(dataset)):
#   temp = dataset[i][0].shape[2]
#   if temp is not 100:
#     print("{} : {}".format(i, dataset[i][0].shape))

