import pytorch_lightning as pl


class FaceTrackDataModule(pl.LightningDataModule):
    """Input paths to folders that contain images and corresponding annotations.

    """

    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        self.mnist_test = MNIST(self.data_dir, train=False)
        mnist_full = MNIST(self.data_dir, train=True)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)