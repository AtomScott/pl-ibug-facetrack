import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import FaceTrackDatasetFolder
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch.utils.data.sampler import BatchSampler, SequentialSampler, Sampler

from tqdm import tqdm
class MyBatchSampler(BatchSampler):
    def __init__(
        self, sampler, batch_size: int, drop_last: bool, chunk_lens: list
    ) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self.chunk_lens = chunk_lens

    def __iter__(self):
        chunk_lens = self.chunk_lens
        offset = 0
        for chunk_len in chunk_lens:
            batch = []
            for idx in range(offset, offset + chunk_len):  # self.sampler: # range(37)
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                yield batch
            offset += chunk_len

    def __len__(self):
        chunk_lens = self.chunk_lens
        if self.drop_last:
            l = sum([chunk_len // self.batch_size for chunk_len in chunk_lens])
            return l  # type: ignore
        else:
            return sum([(chunk_len + self.batch_size - 1) // self.batch_size for chunk_len in chunk_lens])  # type: ignore


class FaceTrackDataModule(pl.LightningDataModule):
    """Input paths to folders that contain images and corresponding annotations.

    """

    def __init__(self, paths, batch_size, n_workers):
        super().__init__()
        self.paths = paths
        self.batch_size = batch_size
        self.n_workers = n_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_paths, val_paths = train_test_split(
            self.paths, train_size=0.7, random_state=0
        )

        self.train_chunks = [FaceTrackDatasetFolder(train_path) for train_path in train_paths]
        self.train_ds = ConcatDataset(
            self.train_chunks
        )
        self.val_chunks = [FaceTrackDatasetFolder(val_path) for val_path in val_paths]
        self.val_ds = ConcatDataset(
            self.val_chunks
        )

    def train_dataloader(self):
        chunk_lens = list(map(len, self.train_chunks))
        my_batch_sampler = MyBatchSampler(
            batch_size=self.batch_size,
            sampler=SequentialSampler(self.train_ds),
            chunk_lens=chunk_lens,
            drop_last=False,
        )
        return DataLoader(
            self.train_ds, batch_sampler=my_batch_sampler, num_workers=self.n_workers
        )

    def val_dataloader(self):
        chunk_lens = list(map(len, self.val_chunks))
        my_batch_sampler = MyBatchSampler(
            batch_size=self.batch_size,
            sampler=SequentialSampler(self.val_ds),
            chunk_lens=chunk_lens,
            drop_last=False,
        )

        return DataLoader(
            self.val_ds, batch_sampler=my_batch_sampler, num_workers=self.n_workers
        )


if __name__ == "__main__":

    root = Path("./data/")

    paths = [
        str(p)
        for p in list(root.glob("300VW/*/")) + list(root.glob("CONFER/**/stitched"))
    ]

    print(f"Found n={len(paths)} paths")
    dm = FaceTrackDataModule(paths, batch_size=128, n_workers=6)

    dm.prepare_data()
    dm.setup()

    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()

    for x, y in tqdm(train_dl):
        pass

    for x, y in tqdm(val_dl):
        pass

