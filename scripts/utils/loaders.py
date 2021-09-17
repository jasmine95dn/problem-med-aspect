import torch
import h5py
import pandas as pd


class Loader:

    def __init__(self, input_dir='.', ftype=''):
        self.input_dir = input_dir
        self.ftype = ftype

    def __repr__(self):
        return self.input_dir


class ModelLoader(Loader):

    @classmethod
    def load_model(cls, path, device):
        """

        :param path:
        :param device:
        :return:
        """
        assert (path is not None)
        print(f'Model loaded from <== {path}')
        state_dict = torch.load(path, map_location=device)

        return state_dict


class EmbeddingLoader(Loader):

    @staticmethod
    def load_embed(path):
        """

        :param path:
        :return:
        """
        return h5py.File(path)


class FrameLoader(Loader):

    def __init__(self, input_dir='./', filetype='.json', to_compress=False):
        super(FrameLoader).__init__(input_dir, filetype)
        self.to_compress = to_compress

    @classmethod
    def load_frame(cls, path):
        """

        :param path:
        :return:
        """
        assert (path is not None)
        print(f'Data loaded from <=== {path}')
        # Load the dataset into a pandas dastaframe
        df = pd.read_json(path)

        # Get the lists of sentences and their labels.
        sentences = df.sent.values
        labels = df.label.values

        return sentences, labels

    @classmethod
    def load_infer_frame_cf_matrix(cls, path):
        """

        :param path:
        :return:
        """
        assert (path is not None)
        print(f'Inferred data loaded from <=== {path}')
        # Load the inferred output with predicted and true labels to compare
        df = pd.read_json(path)

        # Extract true and predicted labels
        true = df.true
        pred = df.pred

        return true, pred