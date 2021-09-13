# -*- coding: utf8 -*-
import torch
import h5py
import os
from pathlib import Path
import pandas as pd


class Saver:

    def __init__(self, output_dir, filetype='.txt'):

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.filetype = filetype

    def __repr__(self):
        return self.output_dir


class ModelSaver(Saver):

    @classmethod
    def save_checkpoint(cls, model, optimizer, valid_loss, path):
        """

        :param model:
        :param optimizer:
        :param valid_loss:
        :param path:
        :return:
        """
        assert(path is not None)

        state_dict = {'model_state_dict': model.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                       'valid_loss': valid_loss}
        torch.save(state_dict, path)
        print(f'Model saved to ==> {path}')


class EmbeddingSaver(Saver):

    def save_embed(self, idx, output, filename):
        """

        :param idx:
        :param output:
        :param filename:
        :return:
        """
        path = os.path.join(self.output_dir, f"{filename}.{self.filetype}")
        with h5py.File(path, 'a') as file2write:
            file2write.create_dataset(str(idx), output.shape, dtype='float32', data=output)


class FrameSaver(Saver):

    def __init__(self, output_dir='./', filetype='.json', to_compress=False):

        super(FrameSaver).__init__(output_dir, filetype)
        self.to_compress = to_compress

    def save_input_frame(self, table, filename):
        """

        :param table:
        :param filename:
        :return:
        """
        path = os.path.join(self.output_dir, filename)

        archive_name = f"{path}.{self.filetype}"
        compression_opts = None

        # define only if required to compress output data
        if self.to_compress:
            compression_opts = dict(method='zip', archive_name=archive_name)
            archive_name = f"{path}.zip"

        table.to_json(archive_name, orient='records', compression=compression_opts)

    @classmethod
    def save_infer_frame(cls, pred, true, path):
        """

        :param pred:
        :param true:
        :param path:
        :return:
        """
        output = pd.DataFrame(zip(true.tolist(), pred.argmax(axis=1).tolist()), columns=['pred', 'true'])
        print(f'Predicted and true labels saved into ==> {path}')
        output.to_json(path, orient='records')




