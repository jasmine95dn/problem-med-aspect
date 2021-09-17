# --*-- coding: utf-8 --*--

"""

"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split
import datetime, time, random
from scripts.nn.metrics import accuracy_score, prec_rec_fscore
from .config import ModelConfig, EmbeddingConfig
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings, FlairEmbeddings, StackedEmbeddings, WordEmbeddings
from scripts.utils.loaders import FrameLoader, ModelLoader
from scripts.utils.savers import ModelSaver, FrameSaver
import numpy as np


class EntityEmbeddingProcessor:

    """

    """
    def __init__(self, etype, device, finetune=False):
        """
        Define an embedding processor representing an entity from an input sequence
        :param etype: type of embedding, name on this list:
            - BERT (bert-large-uncased-whole-word-masking)
            - FLAIR (trained on news corpus)
            - ClinicalBERT (emily/alsentzer
        :param device:
        """
        self.etype = etype
        self.embedding1 = None
        self.embedding2 = None
        self.size = 512

        if etype in {'bert', 'clinicalbert', 'biobert'}:
            if finetune:
                self.embedding1 = BertModel.from_pretrained('bert-base-uncased') if etype == 'bert' \
                        else BertModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT') if etype == 'clinicalbert' \
                        else BertModel.from_pretrained('dmslab/BioBERT v1.1')
            else:
                self.embedding1 = BertModel.from_pretrained('bert-large-uncased-whole-word-masking') if etype == 'bert' \
                            else BertModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
            self.size = self.embedding1.config.hidden_size

        elif etype in {'flair', 'hunflair', 'bertflair', 'berthunflair'}:
            flair_embs = [FlairEmbeddings('pubmed-forward'), FlairEmbeddings('pubmed-backward')] \
                        if etype.endswith('hunflair') \
                        else [FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward')]
            emb = TransformerWordEmbeddings('bert-large-uncased-whole-word-masking') if etype.startswith('bert') \
                        else WordEmbeddings('pubmed') if etype.endswith('hunflair') \
                        else WordEmbeddings('glove')

            self.embedding2 = StackedEmbeddings([*flair_embs, emb])
            self.size = self.embedding2.embedding_length

        elif etype in {'clinicalbertflair', 'clinicalberthunflair'}:
            self.embedding1 = BertModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
            self.embedding2 = StackedEmbeddings([FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward')])  \
                            if etype == 'clinicalbertflair' \
                        else StackedEmbeddings([FlairEmbeddings('pubmed-forward'), FlairEmbeddings('pubmed-backward')])
            self.size = self.embedding1.config.hidden_size + self.embedding2.embedding_length

        self.embedding1 = self.embedding1.to(device)
        self.embedding2 = self.embedding2.to(device)

    def __call__(self, input_ids, input_mask=None, tokenizer=None):
        inputs = None
        if self.etype == 'bert' or self.etype == 'clinicalbert':
            emb = self.embedding1(input_ids=input_ids, attention_mask=input_mask)
            inputs = emb[0]

        elif self.etype in {'flair', 'hunflair', 'bertflair', 'berthunflair'}:

            # 1. recreate sentences from b_input_ids in transformers 
            sents = [tokenizer.convert_ids_to_tokens(sent) for sent in input_ids]
            # 2. move sents to Sentence in flair
            tok_sents = [Sentence(s) for s in sents]
            # 3. apply embedding into tok_sents
            self.embedding2.embed(tok_sents)
            # 4. extract the embeddings only
            inputs = torch.stack([torch.stack([tok.embedding for tok in sent]) for sent in tok_sents])

        elif self.etype in {'clinicalbertflair', 'clinicalberthunflair'}:
            # in transformer
            emb = self.embedding1(input_ids=input_ids, attention_mask=input_mask)
            clinicalbert = emb[0]

            # in flair
            # 1. recreate sentences from b_input_ids in transformers 
            sents = [tokenizer.convert_ids_to_tokens(sent) for sent in input_ids]
            # 2. move sents to Sentence in flair
            tok_sents = [Sentence(s) for s in sents]
            # 3. apply embedding into tok_sents
            self.embedding2.embed(tok_sents)
            # 4. extract the embeddings only
            flair = torch.stack([torch.stack([tok.embedding for tok in sent]) for sent in tok_sents])

            # stacking clinicalbert and flair
            inputs = torch.cat([clinicalbert, flair], dim=-1)

        return inputs


class ModelProcessor:

    def __init__(self, model_config: ModelConfig, model, embedding_config: EmbeddingConfig):
        """

        :param model_config:
        :param model:
        :param embedding_config:
        """

        # data
        self.tokenizer = BertTokenizer.from_pretrained(model_config.model_path,
                                                       do_lower_case=True, never_spit=embedding_config.never_split)
        train_sents, train_labels = FrameLoader.load_frame(model_config.train_path)
        train_dataset, val_dataset = self.split_train(train_data=self.tokenize(train_sents, train_labels),
                                                        prop=model_config.proportion)
        test_sents, test_labels = FrameLoader.load_frame(model_config.test_path)
        test_dataset = self.tokenize(model_config.model_path, test_sents, test_labels, embedding_config.never_split)

        self.train_loader = self.data_loader(train_dataset, sampler=RandomSampler, batch_size=model_config.batch_size)
        self.valid_loader = self.data_loader(val_dataset, sampler=SequentialSampler, batch_size=model_config.batch_size)
        self.test_loader = self.data_loader(test_dataset, sampler=SequentialSampler, batch_size=model_config.batch_size)

        # define embedding
        self.embedding = EntityEmbeddingProcessor(etype=embedding_config.etype, device=embedding_config.device)
        self.emb_type = embedding_config.etype

        # define model
        self.model = model(input_size=self.embedding.size, hidden_size=model_config.hidden_size,
                           num_labels=model_config.num_classes, num_layers=model_config.num_layers,
                           dropout=model_config.drop_prob) if not model_config.finetune \
            else model(model_config.model_path, model_config.input_size, model_config.hidden_size,
                         model_config.num_classes, model_config.freeze_bert)
        self.finetune = model_config.finetune

        # model path to continue training
        self.train_model_path = model_config.train_model_path

        # Total number of training steps is [number of batches] x [number of epochs].
        # (Note that this is not the same as the number of training samples).
        total_steps = len(self.train_loader) * model_config.num_epochs

        # define loss, optimizer and scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=model_config.l_rate, eps=model_config.eps)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0,
                                                         num_training_steps=total_steps)

        # output path
        self.model_save_path = model_config.model_save_path
        self.infer_save_path = model_config.infer_save_path

        # set device
        self.device = model_config.device

        # set num_epochs
        self.num_epochs = model_config.num_epochs

    def tokenize(self, data, labels):
        """

        :param data:
        :param labels:
        :return:
        """
        encoded_dict = self.tokenizer(data.tolist(), padding=True, truncation=True, return_tensors='pt')

        # get sentences
        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']

        # get labels
        labels = torch.tensor(labels)

        # Combine the training inputs into a TensorDataset.
        dataset = TensorDataset(input_ids, attention_masks, labels)

        return dataset

    @staticmethod
    def split_train(train_data:TensorDataset, prop: float = 0.9):
        """

        :param train_data:
        :param prop:
        :return:
        """
        train_size = int(prop * len(train_data))
        val_size = len(train_data) - train_size

        train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

        return train_dataset, val_dataset


    @staticmethod
    def data_loader(dataset: TensorDataset, sampler, batch_size: int = 8) -> DataLoader:
        """
        speed of training by dividing and process data in batches
        :param dataset:
        :param sampler:
        :param batch_size:
        :return:
        """
        return DataLoader(dataset,  # data samples
                          sampler=sampler(dataset),  # select batch
                          batch_size=batch_size)

    @staticmethod
    def format_time(elapsed):
        """
        Takes a time in seconds and returns a string hh:mm:ss
        :param elapsed:
        :return:
        """
        # Round to the nearest second.
        elapsed_rounded = int(round(elapsed))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def return_logits(self, input_ids, input_mask, finetune=False):
        if finetune:
            logits = self.model(input_ids, input_mask)
        else:
            # create embedding
            inputs = self.embedding(input_ids, input_mask)
            logits = self.model(inputs)
        return logits

    def train(self, dataloader, optimizer, criterion, scheduler, t0):
        """

        :param dataloader:
        :param optimizer:
        :param criterion:
        :param scheduler:
        :param t0:
        :return:
        """
        # Reset the total loss for this epoch.
        total_train_loss = 0
        total_train_prec = 0
        total_train_rec = 0
        total_train_f1 = 0

        # model in train mode
        self.model.train()

        # For each batch of training data...
        for step, batch in enumerate(dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = self.format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(dataloader), elapsed))

            # Unpack this training batch from our dataloader. 
            # Load batch to GPU
            b_input_ids, b_input_mask, b_labels = tuple(t.to(self.device) for t in batch)

            # Always clear any previously calculated gradients before performing a
            # backward pass. reset the gradients after every batch
            optimizer.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return logits
            logits = self.return_logits(b_input_ids, b_input_mask, self.finetune)

            loss = criterion(logits, b_labels)
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

            # Move logits and labels to CPU
            logits = logits.to('cpu')
            label_ids = b_labels.to('cpu')

            # Calculate metrics
            prec, rec, f1 = prec_rec_fscore(label_ids, logits)

            total_train_prec += prec
            total_train_rec += rec
            total_train_f1 += f1

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(dataloader)
        avg_train_prec = total_train_prec / len(dataloader)
        avg_train_rec = total_train_rec / len(dataloader)
        avg_train_f1 = total_train_f1 / len(dataloader)

        return avg_train_loss, avg_train_prec, avg_train_rec, avg_train_f1

    def evaluate(self, dataloader, criterion):
        """

        :param dataloader:
        :param criterion:
        :return:
        """
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        self.model.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_prec = 0
        total_eval_rec = 0
        total_eval_f1 = 0
        total_eval_loss = 0

        # Evaluate data for one epoch
        for batch in dataloader:

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, and load to the GPU using 
            # the `to` method.
            b_input_ids, b_input_mask, b_labels = tuple(t.to(self.device) for t in batch)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():

                # Forward pass, calculate logit predictions.
                logits = self.return_logits(b_input_ids, b_input_mask, self.finetune)

            # Compute loss
            loss = criterion(logits, b_labels)

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.to('cpu')
            label_ids = b_labels.to('cpu')

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += accuracy_score(label_ids, logits)

            # Calculate precision, recall, f1
            prec, rec, f1 = prec_rec_fscore(label_ids, logits)

            total_eval_prec += prec
            total_eval_rec += rec
            total_eval_f1 += f1

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(dataloader)
        avg_eval_prec = total_eval_prec / len(dataloader)
        avg_eval_rec = total_eval_rec / len(dataloader)
        avg_eval_f1 = total_eval_f1 / len(dataloader)

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(dataloader)

        return avg_val_accuracy, avg_eval_prec, avg_eval_rec, avg_eval_f1, avg_val_loss

    def test(self, dataloader):
        """

        :param dataloader:
        :return:
        """
        # Put model in evaluation mode
        self.model.eval()

        # Tracking variables 
        predictions , true_labels = [], []

        # Predict 
        for batch in dataloader:

            # Unpack the inputs from our dataloader and add to GPU
            b_input_ids, b_input_mask, b_labels = tuple(t.to(self.device) for t in batch)

            # Telling the model not to compute or store gradients, saving memory and 
            # speeding up prediction
            with torch.no_grad():
                # create embedding
                inputs = self.embedding(input_ids=b_input_ids, attention_mask=b_input_mask)
                # Forward pass, calculate logit predictions.
                logits = self.model(inputs)

            # Move logits and labels to CPU
            logits = logits.to('cpu')
            label_ids = b_labels.to('cpu')

            # Store predictions and true labels
            predictions.append(logits)
            true_labels.append(label_ids)

        print('    DONE.')

        predictions = torch.cat(predictions, axis=0)
        true_labels = torch.cat(true_labels, axis=0)

        return predictions, true_labels

    def __call__(self, mode='train', cond=False):
        """

        :param mode:
        :param cond:
        :return:
        """

        if mode == 'train':

            # set seed value for reproducibility
            seed_val = 42
            random.seed(seed_val)
            np.random.seed(seed_val)
            torch.manual_seed(seed_val)
            torch.cuda.manual_seed_all(seed_val)

            # define parameters for valid loss
            best_valid_loss = float('inf')
            n_unchanged = 0
            last_val_loss = 0

            # set epoch start
            epoch_start = 1

            # if continue the training and not from the start
            if cond:
                state_dict = ModelLoader.load_model(self.train_model_path, self.device)
                epoch_start = int(self.train_model_path[-4])
                self.model.load_state_dict(state_dict['model_state_dict'])
                self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

            # Measure the total training time for the whole run.
            total_t0 = time.time()

            # For each epoch...
            for epoch_i in range(epoch_start, self.num_epochs+1):

                # ========================================
                #               Training
                # ========================================

                # Perform one full pass over the training set.

                print("")
                print(f'======== Epoch {epoch_i:} / {self.num_epochs:} ========')
                print('Training...')

                # Measure how long the training epoch takes.
                t0 = time.time()

                # train the model
                avg_train_loss, avg_train_prec, avg_train_rec, avg_train_f1 = self.train(
                                                self.train_loader, self.optimizer, self.criterion, self.scheduler, t0)

                # Measure how long this epoch took.
                training_time = self.format_time(time.time() - t0)

                print("")
                print("  Average training loss: {0:.2f}".format(avg_train_loss))
                print(f""" Average Precision: {avg_train_prec:.1f}, 
                           Average Recall: {avg_train_rec:.1f}, Average F1: {avg_train_f1:.1f}""")
                print("  Training epoch took: {:}".format(training_time))

                # ========================================
                #               Validation
                # ========================================
                # After the completion of each training epoch, measure our performance on
                # our validation set.

                print("")
                print("Running Validation...")

                t0 = time.time()

                # evaluate the model
                avg_val_accuracy, avg_eval_prec, avg_eval_rec, avg_eval_f1, avg_val_loss = self.evaluate(self.model,
                                                                self.embedding, self.valid_loader, self.criterion)

                # Measure how long the validation run took.
                validation_time = self.format_time(time.time() - t0)
                print(f" Accuracy: {avg_val_accuracy:.2f}")
                print(f"""" Average Precision: {avg_eval_prec:.1f}, 
                            Average Recall: {avg_eval_rec:.1f}, Average F1: {avg_eval_f1:.1f}""")
                print(f" Validation Loss: {avg_val_loss:.2f}")
                print(f"  Validation took: {validation_time:}")

                # save the best model
                if avg_val_loss < best_valid_loss:
                    best_valid_loss = avg_val_loss

                    ModelSaver.save_checkpoint(self.model, self.optimizer, avg_val_loss,
                                        path=f'{self.model_save_path}/{self.emb_type}_saved_weight_epoch{epoch_i}.pt')

                # check if loss stays unchanged
                if last_val_loss == avg_val_loss:
                    n_unchanged += 1
                else:
                    last_val_loss = avg_val_loss
                    n_unchanged = 0
                # val loss stays unchanged for 5 consecutive epochs then stops training
                if n_unchanged == 5:
                    print(f" Stop training at epoch {epoch_i}")
                    break

            print("")
            print("Training complete!")

            print("Total training took {:} (h:mm:ss)".format(self.format_time(time.time()-total_t0)))

            ModelSaver.save_checkpoint(self.model, self.optimizer, last_val_loss,
                                       path=f'{self.model_save_path}/{self.emb_type}_model.pt')

        elif mode == 'test':
            # load model to test
            state_dict = ModelLoader.load_model(self.train_model_path, self.device)
            self.model.load_state_dict(state_dict['model_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

            print(f'Predicting labels for {len(self.test_loader):,} test sentences...')
            # get predictions and labels
            predictions, true_labels = self.test(self.test_loader)

            # calculate metrics
            prec, rec, f1 = prec_rec_fscore(true_labels, predictions)
            print(f" Precision: {prec:.1f}, Recall: {rec:.1f}, F1: {f1:.1f}")

            acc = accuracy_score(true_labels, predictions)
            print(f"  Accuracy: {acc:.2f}")

            # save for inference
            FrameSaver.save_infer_frame(predictions, true_labels, f'{self.infer_save_path}/{self.emb_type}_infer.json')









