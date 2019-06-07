from train_module import Options, Trainer

opt = Options().parse()
trainer = Trainer(opt)
trainer.train()


# from dataset import MyDataset

# from model import vgg11_bn
# from torch.utils.data import DataLoader

# dataset_save_path = 'dataset.pkl'
# dataset_path = "../speech_data/"

# dataset = MyDataset(dataset_path)


# for i in range(len(dataset)):
#   temp = dataset[i][0].shape[2]
#   if temp is not 100:
#     print("{} : {}".format(i, dataset[i][0].shape))

