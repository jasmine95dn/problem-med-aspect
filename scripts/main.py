# -*- coding: utf8 -*-
import argparse
import sys
import scripts.utils.commanders as commanders
from scripts.nn.config import PrepConfig, ModelConfig, EmbeddingConfig


class Problem(object):

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='''Full scripts for whole project''',
            usage='''main.py <command> [<args>]
            
            Commands for this script:
            prep    Run prepocessing
            mod     Run processor
            eval    Run evaluator
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
        parser.add_argument('-o', '--output', type=str, default='./', help='output folder')
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
        else: tags = args.ttags
        config = PrepConfig(srcfolder=args.input, outfolder=args.output, dataset_type=args.dtype,
                            to_compress=args.compress, raw_pref=args.raw_pref,
                            temprel_tags=tags, window_size=args.window_size)
        # run
        commanders.run_prep(config)

    def mod(self):
        parser = argparse.ArgumentParser(description='Run model')

        # model configurations
        parser.add_argument('train', type=str,  help='input training file, expected json file')
        parser.add_argument('test', type=str, help='input test file, expected json file')
        parser.add_argument('--labels', type=list)

        parser.add_argument('-o', '--output', type=str, default='new_model.pt', help='output file')
        parser.add_argument('-t', '--task', type=str, default='modpol', help='task name')
        parser.add_argument('--sent_status', type=str, default='within',
                            help='status for classifier in rela-task (default: within)')
        parser.add_argument('--mode', type=str, default='train',
                            help='which mode to run this model, train or test (default: train)')

        # hyperparameter
        parser.add_argument('-n', '--n_layers', type=int, default=3,
                            help='''number of hidden layers of classifier (default: 3 layers)''')

        parser.add_argument('-lr', '--l_rate', type=float, default=0.0001,
                            help='''learning rate (default: 0.0001''')
        parser.add_argument('-opt', '--optim', type=str, default='Adam',
                            help='''optimizer (default: Adam)''')
        # todo: dropout, hidden size, batch size,
        parser.add_argument('--dropout')
        parser.add_argument('-hs', '--hidden_size', type=int, default=256, help='hidden size')
        parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')

        # training mode

        # embedding config
        parser.add_argument('-et', '--etype', type=str, default='bert', help='embedding type')
        parser.add_argument('--never_split', type=list, default=None,
                        help='tokens that are not split by WordPieceTokenizer from BERTTokenizer, default: <p>, </p>')

        args = parser.parse_args(sys.argv[2:])

        # define embedding configuration
        embedding_config = EmbeddingConfig(etype=args.etype, never_split=args.never_split)

        # define model configuration
        model_config = ModelConfig(
                train_path=args.train, test_path=args.test, labels=[0, 1, 2], proportion=0.9,  # raw data
                model_path='bert-base-uncased',  # tokenizer
                train_model_path='model.pt',  # in case of continue the training from old training
                batch_size= 500, hidden_size= 512, num_layers= 2, num_epochs = 10,
                drop_prob = 0.2,  # train hyperparameters
                freeze_bert= False, finetune=False, bert_hidden_size=768,  # type of model
                l_rate= 0.001, eps= 1e-8,  # optimizer hyperparameters
                model_save_path='out/model', infer_save_path='out/infer')

        # todo: run_mod
        commanders.run_mod(model_config, embedding_config,
                           mod=args.task, sent_state=args.sent_status, mode=args.mode, info='start')

    def eval(self):
        parser = argparse.ArgumentParser(description='Run evaluation')
        parser.add_argument()
        args = parser.parse_args(sys.argv[2:])
        
        # todo: run_eval
        commanders.run_eval(args)

    def plot(self):
        parser = argparse.ArgumentParser(description='Run plot')
        parser.add_argument()
        args = parser.parse_args(sys.argv[2:])
        
        # todo: run_plot
        commanders.run_plot(args)


if __name__ == '__main__':
    Problem()





