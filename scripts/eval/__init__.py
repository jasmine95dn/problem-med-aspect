# -*- coding : utf8 -*-

from scripts.eval.plotters import cf_matrix, cf_matrix_plot
from scripts.nn.config import PlotConfig
from scripts.utils.loaders import FrameLoader


def plot_cf(config: PlotConfig):
    true, pred = FrameLoader.load_infer_frame_cf_matrix(path=config.infer_path)

    # create matrix
    matrix = cf_matrix(y_true=true, y_predict=pred, labels=config.labels, error=config.error)

    # plot confusion matrix
    cf_matrix_plot(matrix=matrix, labels=config.labels, size=config.figsize,
                   save=config.to_save, outdir=config.output_path, err=config.error)