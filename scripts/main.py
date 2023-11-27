# -*- coding: utf8 -*-
import argparse
import sys
import scripts.utils.commanders as commanders
from scripts.nn.config import PrepConfig, ModelConfig, EmbeddingConfig, PlotConfig


class Problem(object):

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='''Full scripts for whole project''',
            usage='''main.py <command> [<args>]
            
            Commands for this script:
            prep    Run prepocessing
            mod     Run processor
            plot    Run plotter''')

        parser.add_argument('command', help='Subcommand to run')
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unknown command')
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def prep(self):
        parser = argparse.ArgumentParser(description='Run preprocessing')
        parser.add_argument('input', type=str, help='input file/directory')
        parser.add_argument('-o', '--output', type=str, default='./', help='output folder (default: current directory)')
        parser.add_argument('-dt', '--dtype', type=str, default='train',
                            help='''data type (default: train data), choose train/test''')
        parser.add_argument('--compress', action='store_true', help='whether to compress output file')
        parser.add_argument('--raw_pref', type=str, default='',
                            help='preference in raw text, 2010 is empty (default), 2012 is .xml')
        parser.add_argument('--ttags', type=str, nargs=4, default=None,
                            help='list of tags in form [2 xml tags, 2 nonxml tags]')
        parser.add_argument('-ws', '--window_size', type=int, default=1500, help='context window size')
        args = parser.parse_args(sys.argv[2:])

        # define configurations
        if args.ttags:
            tags = {'xml': args.ttags[:2], 'nonxml': args.ttags[2:]}
        else:
            tags = args.ttags
        config = PrepConfig(srcfolder=args.input, outfolder=args.output, dataset_type=args.dtype,
                            to_compress=args.compress, raw_pref=args.raw_pref,
                            temprel_tags=tags, window_size=args.window_size)
        # run
        commanders.run_prep(config)

    def mod(self):
        parser = argparse.ArgumentParser(description='Run model')

        # model configurations
        # 1. data
        parser.add_argument('train', type=str,  help='input training file, expected json file')
        parser.add_argument('test', type=str, help='input test file, expected json file')
        parser.add_argument('--labels', type=int, nargs='+', default=[0, 1, 2],
                            help='list of labels (default: [0,1,2])')
        parser.add_argument('--train_prop', type=float, default=0.9,
                            help='train data proportion from whole training set (default: 0.9)')
        # 2. tokenizer
        parser.add_argument('--model_path', type=str, default='bert-base-uncased',
                            help='transformers BERT tokenizer model (default: bert-base-uncased)')

        # 3. in case of continuing the training from saved model
        parser.add_argument('--train_model_path', type=str, default=None,
                            help='continue the training process from which checkpoint, input is a .pt file')

        # 4. model type freeze_bert=False, finetune=False, bert_hidden_size=768
        parser.add_argument('--finetune', action='store_true',
                            help='call this flag to call finetuned model instead of RNN')
        parser.add_argument('-bhs', '--bert_hidden_size', type=int, default=768,
                            help='BERT hidden size as input (default: 768)')
        parser.add_argument('--freeze_bert', action='store_true',
                            help='call this flag in testing mode from finetuning')

        # 5. train hyperparameter batch_size=500, hidden_size=512, num_layers=2, num_epochs=10, drop_prob=0.2
        # l_rate=0.001, eps=1e-8,
        parser.add_argument('-hs', '--hidden_size', type=int, default=512, help='hidden size')
        parser.add_argument('-b', '--batch_size', type=int, default=8, help='batch size')
        parser.add_argument('-n', '--num_layers', type=int, default=2,
                            help='''number of hidden layers of RNN classifier (default: 2 layers)''')
        parser.add_argument('-n_epochs', type=int, help='number of training epochs (default: 10 epochs)')

        parser.add_argument('-lr', '--l_rate', type=float, default=0.001,
                            help='''learning rate (default: 0.001''')
        parser.add_argument('--eps', type=float, default=1e-8,
                            help='''AdamW optimizer epsilon (default: 1e-8)''')
        parser.add_argument('--dropout', type=float, default=0.2, help='RNN dropout rate')

        # 6. outputs
        parser.add_argument('-o', '--output', type=str, default='out/model',
                            help='output path for model saved, only prefix, no definition of ending file')
        parser.add_argument('-i', '--infer_output', type=str, default='out/infer',
                            help='''output inference path to save true and predicted labels in testing, 
                            only prefix, no definition of ending file''')

        # embedding configurations
        parser.add_argument('-et', '--etype', type=str, default='bert', help='embedding type')
        parser.add_argument('--never_split', type=list, default=None,
                        help='tokens that are not split by WordPieceTokenizer from BERTTokenizer, default: <p>, </p>')

        # direct specification for training model
        parser.add_argument('-t', '--task', type=str, default='modpol', help='task name')
        parser.add_argument('--sent_status', type=str, default='within',
                            help='status for classifier in rela-task (default: within)')
        parser.add_argument('--mode', type=str, default='train',
                            help='which mode to run this model, train or test (default: train)')
        parser.add_argument('--cond', type=str, action='store_true',
                            help='''call this flag to inform whether continue from pretrained RNN model, 
                                    only work with flag --mode and have to specify with flag --train_model_path 
                                    (default: off, training from the beginning)''')

        args = parser.parse_args(sys.argv[2:])

        # define embedding configuration
        embedding_config = EmbeddingConfig(etype=args.etype, never_split=args.never_split)

        # check flags
        # 1. if --cond is on, --train_model_path must be specified
        if args.cond and not args.train_model_path:
            sys.stderr.write('No pretrained model is specified')
            sys.exit(1)

        # define model configuration
        model_config = ModelConfig(
                train_path=args.train, test_path=args.test, labels=args.labels, proportion=args.train_prop,  # raw data
                model_path=args.model_path,  # tokenizer
                train_model_path=args.train_model_path,  # in case of continue the training from old training
                batch_size=args.batch_size, hidden_size=args.hidden_size, num_layers=args.num_layers,
                num_epochs=args.num_epochs, drop_prob=args.dropout,  # train hyperparameters
                freeze_bert=args.freeze_bert, finetune=args.finetune,
                bert_hidden_size=args.bert_hidden_size,  # type of model
                l_rate=args.l_rate, eps=args.eps,  # optimizer hyperparameters
                model_save_path=args.output, infer_save_path=args.infer_output)

        commanders.run_mod(model_config, embedding_config,
                           mod=args.task, sent_state=args.sent_status, mode=args.mode, cond=args.info)

    def plot(self):
        parser = argparse.ArgumentParser(description='Run plot')
        parser.add_argument('input', type=str, help='input infererred .json file to plot')
        parser.add_argument('--output', type=str, default='./', help='output path to save (default :./)')
        parser.add_argument('--ptype', type=str, default='cf_matrix', help='type of plot to use (default: cf_matrix)')
        parser.add_argument('--noerror', action='store_false',
                            help='if not called, then cf matrix returned is for the wrong labeling matrix')
        parser.add_argument('--labels', type=int, nargs='+', default=[0,1,2], help='labels (default: [0,1,2])')
        parser.add_argument('--size', type=int, nargs=2, default=(5,5), help='plot size')
        parser.add_argument('--show_only', action='store_true',
                            help='whether to show this plot only, flag not called then figure will be saved')
        args = parser.parse_args(sys.argv[2:])

        config = PlotConfig(infer_path=args.input, output_path=args.output, ptype=args.ptype, error=args.noerror,
                            labels=args.labels, figsize=args.size, to_save=args.show_only)
        # run
        commanders.run_plot(config)


if __name__ == '__main__':
    Problem()





