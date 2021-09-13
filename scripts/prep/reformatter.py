# --*-- coding: utf-8 --*--
import os, re
from itertools import chain
import pandas as pd
import warnings
from scripts.utils.savers import FrameSaver
from scripts.nn.config import PrepConfig
warnings.filterwarnings("ignore")

class Reformat:
    def __init__(self, prep_config: PrepConfig):
        """

        :param prep_config:
        """
        self.srcfolder = prep_config.srcfolder
        self.docs = self.doc_list(prep_config.srcfolder)
        self.data_type = prep_config.dataset_type
        self.rawtext = self.extract_rawtext(self.docs, prep_config.srcfolder, prep_config.raw_pref)
        self.saver = FrameSaver(prep_config.outfolder, to_compress=prep_config.to_compress)

    @staticmethod
    def doc_list(srcfolder, name=r'\d+'):
        """

        :param srcfolder:
        :param name:
        :return:
        """
        docs = set(re.search(name, f).group() for f in os.listdir(srcfolder))

        return docs

    @staticmethod
    def load_raw_text(srcfile):
        """

        :param srcfile:
        :return:
        """
        with open(srcfile) as src:
            rawtxt = {i: {pos: token for pos, token in enumerate(line.strip().split())} for i, line in
                      enumerate(src.readlines(), start=1)}

        return rawtxt

    def extract_rawtext(self, docs, folder, pref):
        """

        :param docs:
        :param folder:
        :param pref:
        :return:
        """
        rawtexts = {doc_id: self.load_raw_text(os.path.join(folder, f'{doc_id}{pref}.txt'))
                    for doc_id in docs}
        return rawtexts

    @staticmethod
    def filter_out_value(df, column, upper=True, match=r'="(.+)"$'):
        """

        :param df:
        :param column:
        :param upper:
        :param match:
        :return:
        """
        table = df.copy(deep=True)
        table[column] = table[column].str.extract(match)
        if upper:
            table[column] = table[column].str.upper()
        return table

    @staticmethod
    def split_col_right(col, deli=' ', maxsplit=2):
        """

        :param col:
        :param deli:
        :param maxsplit:
        :return:
        """
        return pd.Series(str(col).rsplit(deli, maxsplit=maxsplit))

    @staticmethod
    def split_col(col, deli=':'):
        """

        :param col:
        :param deli:
        :return:
        """
        return pd.Series(str(col).split(deli))

    def find_context(self, sent_ind, start, end, doc_id, full=True):
        """

        :param sent_ind:
        :param start:
        :param end:
        :param doc_id:
        :param full:
        :return:
        """
    
        text = self.rawtext[str(doc_id)]

        left_ctxt_sent = [text[sent_ind][i] for i in range(start)]
        right_ctxt_sent = [text[sent_ind][i] for i in range(end+1, len(text[sent_ind]))]
        if full:
            left_ctxt_cross = [list(text[j].values()) for j in range(1, sent_ind)] + [left_ctxt_sent]
            right_ctxt_cross = [right_ctxt_sent] + [list(text[j].values()) for j in range(sent_ind+1, len(text)+1)]
                
            return left_ctxt_sent, right_ctxt_sent, left_ctxt_cross, right_ctxt_cross

        return left_ctxt_sent, right_ctxt_sent

    @staticmethod
    def split_cols(df, split_list):
        """

        :param df:
        :param split_list:
        :return:
        """

        table = df.copy(deep=True)

        for cols, col, func in split_list:
            table[cols] = table[col].apply(func)

        return table

    def create_split_list(self, pref=None):
        """
        Create a list of splitting to split:
            1. column 'event' into 3 columns 'event', 'pos' (start_pos in sent), 'pos2' (end_pos in sent)
            2. column 'pos' into 2 columns 'sent_ind' (sent_ind in doc), 'start' (start_pos in sent_ind)
            3. column 'pos2' into 2 columns 'sent_ind' (sent_ind in doc), 'end' (end_pos in sent_ind) 
            4. column 'other_event' into 3 columns 'other_event', 'other_pos' (start_pos in sent), 'other_pos2' (end_pos in sent)
            5. column 'other_pos' into 2 columns 'other_sent_ind' (sent_ind in doc), 'other_start' (start_pos in sent_ind)
            6. column 'other_pos2' into 2 columns 'other_sent_ind' (sent_ind in doc), 'other_end' (end_pos in sent_ind)

        """
        split_list = [(['event', 'pos', 'pos2'], 'event', self.split_col_right),
                      (['sent_ind', 'start'], 'pos', self.split_col),
                      (['sent_ind', 'end'], 'pos2', self.split_col)
                     ]
        if pref:
            new_split_list = [([f'{pref}_event', f'{pref}_pos', f'{pref}_pos2'], f'{pref}_event', self.split_col_right),
                              ([f'{pref}_sent_ind', f'{pref}_start'], f'{pref}_pos', self.split_col),
                              ([f'{pref}_sent_ind', f'{pref}_end'], f'{pref}_pos2', self.split_col)
                             ]
            split_list.extend(new_split_list)
                
        return split_list

    def __call__(self):
        raise NotImplementedError


class Reformat2010(Reformat):

    def __init__(self, prep_config: PrepConfig):
        super(Reformat2010, self).__init__(prep_config)

    def extract_concept(self, path_to_concepts):
        
        # get all concepts
        concepts = pd.read_csv(path_to_concepts, sep='\|\|', names=["concept", "ctype"], engine='python')
        
        # clean data in second column: 'ctype'
        concepts = self.filter_out_value(concepts, 'ctype', upper=False)
        
        # get only the row where ctype == problem
        problems = concepts[concepts['ctype'] == 'problem'].reset_index(drop=True)
        
        return problems

    def extract_ast(self, path_to_ast):
        
        # get all assertions
        problems = pd.read_csv(path_to_ast, sep='\|\|', names=["concept", "ctype", "atype"], engine='python')
        
        # clean data in 2 last columns: 'ctype', 'atype'
        for col in list(problems.columns)[1:]:
            problems = self.filter_out_value(problems, col, upper=False)
          
        # create 'modality' column
        problems['modality'] = problems['atype']
        
        # convert and merge ast -> modality
        # convert 'present' + 'absent' -> FACTUAL
        problems['modality'] = problems.modality.replace(['present', 'absent'], 'FACTUAL')
        
        # convert the rest -> NONFACTUAL
        problems['modality'] = problems.modality.replace(r'^(?!.*FACTUAL).*$', 'NONFACTUAL', regex=True)
        
        return problems

    def extract_table(self, path_to_concepts, path_to_ast, doc_id):
        print(f'extract document {doc_id}')
        
        # 1. PROBLEM tag
        concepts = self.extract_concept(path_to_concepts)
        
        # 2. ASSERTION tag
        asts = self.extract_ast(path_to_ast)
        
        # 3. MERGING
        problems = pd.merge(concepts, asts, on=['concept', 'ctype'])
        
        # 4. ASSIGN doc id
        problems['doc_id'] = doc_id
        
        return problems

    def get_table(self, docs, folder):
        """

        :param docs:
        :param folder:
        :return:
        """
        # extract info from each document and concatenate them together in one big table
        problems = pd.concat([self.extract_table(os.path.join(folder, f'{doc_id}.con'),
                                                 os.path.join(folder, f'{doc_id}.ast'),
                                                 doc_id)
                                   for doc_id in docs], ignore_index=True)

        # create split lists
        print('create split lists')
        split_list = self.create_split_list()
        
        # split columns
        print("split columns in full problem table")
        table = self.split_cols(problems, split_list)
        
        # filter out value
        print("filter problem value")
        table = self.filter_out_value(table, 'concept', upper=False)
        
        full_table = table[['doc_id', 'concept', 'modality', 'sent_ind', 'start', 'end']]
        
        # sort table
        print("sort table")
        full_table.sort_values(by='doc_id', ignore_index=True, inplace=True)
        
        # convert numeric values
        print("convert numeric values in columns")
        full_table[["sent_ind", "start", "end"]] = full_table[["sent_ind", "start", "end"]].apply(pd.to_numeric)

        return full_table

    def assign_context(self, df, full=False):
        """
        assign context for table problem
        :param df:
        :param full:
        :return:
        """

        problems = df.copy(deep=True)

        problems[['left', 'right']] = problems.apply(
            lambda row: self.find_context(row['sent_ind'], row['start'], row['end'], row['doc_id'], full=full), axis=1,
            result_type='expand')
        return problems        

    @staticmethod
    def extract_mod(df):
        """

        :param df:
        :return:
        """     
        # save mod table
        print('Extract MODALITY table')
        problem_table = df[["event", "doc_id", "sent_ind", "start", "end", "modality", "left", "right"]]
        problem_table["left"] = problem_table.left.apply(lambda x: ' '.join(x))
        problem_table["right"] = problem_table.right.apply(lambda x: ' '.join(x))
        # combine sent
        problem_table["sent"] = problem_table.left + ' <p> ' + problem_table.event + ' </p> ' + problem_table.right

        return problem_table

    def __call__(self):
        """

        :return:
        """
        # GET TABLE problem
        probs = self.get_table(self.docs, self.srcfolder)

        # ASSIGN CONTEXT TO PROBLEMS (within and cross)
        problems = self.assign_context(probs, full=False)

        # GET AND SAVE PROBLEMS TABLES WITH CONTEXTS mod and pol
        mod_table = self.extract_mod(problems)

        print('Save MODALITY table')
        self.saver.save_input_frame(mod_table, f'problems_{self.data_type}_data_2010_mod')


class Reformat2012(Reformat):

    def __init__(self, prep_config: PrepConfig):
        """

        :param prep_config:
        """
        super(Reformat2012, self).__init__(prep_config)
        self.tags = prep_config.temprel_tags
        self.window_size = prep_config.window_size

    def extract_tlink(self, path_to_tlinks):
        """
        TLINK tag
        :param path_to_tlinks:
        :return:
        """

        # get all temporal relation of tag TLINK
        alltlinks = pd.read_csv(path_to_tlinks, sep=r'\|\|', names=["event1", "event2", "reltype"], engine='python')

        # get only the rows where there is a relation of an EVENT
        tlinks = alltlinks.loc[
            (alltlinks.event1.str.startswith('EVENT')) &
            (alltlinks.event2.str.startswith('EVENT'))].reset_index(drop=True)

        # clean data in column "reltype"
        tlinks = self.filter_out_value(df=tlinks, column='reltype')

        # merge temporal relation labels into 3 main tags: OVERLAP, BEFORE, AFTER
        pairs = [(['SIMULTANEOUS', 'DURING'], 'OVERLAP'),
                 (['BEFORE_OVERLAP', 'ENDED_BY'], 'BEFORE'),
                 ('BEGUN_BY', 'AFTER')]
        for search, repl in pairs:
            tlinks['reltype'].replace(search, repl, inplace=True)

        # we need 2 tables of TLINK, because each pair of relation appears once, an event PROBLEM
        # might occur in either column 'event1' or 'event2'

        # create table of TLINK for events PROBLEM in column 'event1'
        # rename 2 columns 'event1', 'event2'
        tlinks1 = tlinks.rename(columns={"event1": "event", "event2": "other_event"})

        # create table of TLINK for events PROBLEM in column 'event2'
        # rename 2 columns 'event1', 'event2'
        # in this table, the 'reltype' of tags BEFORE and AFTER must be swapped with each other
        tlinks2 = tlinks.rename(columns={"event2": "event", "event1": "other_event"})
        tlinks2.reltype.replace(['BEFORE', 'AFTER'], ['AFTER', 'BEFORE'], inplace=True)

        return tlinks1, tlinks2

    def extract_event(self, path_to_events):
        """
        EVENT tag
        :param path_to_events:
        :return:
        """

        # get all terms including tag of EVENT and TYPE, MODALITY, POLARITY of EVENTs
        all_terms = pd.read_csv(path_to_events, sep=r'\|\|', names=["event", "type", "modality", "polarity"],
                                engine='python')

        # get only the rows where there is tag EVENT
        events = all_terms.loc[all_terms.event.str.startswith('EVENT')]

        # clean data in last 3 columns: 'type', 'modality', 'polarity'
        for col in list(events.columns)[1:]:
            events = self.filter_out_value(events, col)

        # merge modality labels into 2 main tags: FACTUAL, NONFACTUAL
        events['modality'] = events.modality.str.replace(r'^(?!.*FACTUAL).*$', 'NONFACTUAL', regex=True)

        # get only the event PROBLEM
        problems = events[events['type'] == 'PROBLEM']
        problems = problems[['event', 'modality', 'polarity']].reset_index(drop=True)

        return events, problems

    @staticmethod
    def merge(events, problems, *tlinks):
        """
        MERGING
        :param events:
        :param problems:
        :param tlinks:
        :return:
        """

        # concatenate 2 tables and sort ascendingly based on EVENT
        table = pd.concat([pd.merge(left=problems, right=tlink_, on="event") for tlink_ in tlinks], ignore_index=True)
        table.sort_values(by='event', ignore_index=True, inplace=True)

        # assign event 'type' of column 'other_event' into table by merging with table 'events'
        events = events.rename(columns={"event": "other_event"})
        table = pd.merge(table, events, on="other_event")

        # rename columns
        # rename column 'type' -> 'other_event_type'
        table = table.rename(columns={'modality_x': 'event_modality', 'modality_y': 'other_event_modality',
                                      'polarity_x': 'event_polarity', 'polarity_y': 'other_event_polarity',
                                      'type': 'other_event_type'})

        return table

    def extract_table(self, path_to_events, path_to_tlinks, doc_id):
        """

        :param path_to_events:
        :param path_to_tlinks:
        :param doc_id:
        :return:
        """
        print(f"extract document {doc_id}")

        # 1. TLINK tag
        tlinks1, tlinks2 = self.extract_tlink(path_to_tlinks)

        # 2. EVENT tag
        events, problems = self.extract_event(path_to_events)

        # 3. MERGING
        table = self.merge(events, problems, tlinks1, tlinks2)

        # 4. ASSIGN doc id
        table['doc_id'] = doc_id
        problems['doc_id'] = doc_id

        return table, problems

    def add_attr(self, df):
        """

        :param df:
        :return:
        """
        table = df.copy(deep=True)

        # clean data in columns 'event' and 'other_event'
        cols_matches = ['event', 'other_event']
        for col in cols_matches:
            table = self.filter_out_value(table, col, upper=False)

        # assign of temporal relation is 'within' or 'cross' sentence
        table['sent_state'] = 'within'
        table.loc[(table.sent_ind != table.other_sent_ind), 'sent_state'] = 'cross'

        # last table
        table = table.reindex(
            columns=['doc_id', 'event', 'event_modality', 'event_polarity', 'sent_ind', 'start', 'end',
                     'other_event', 'other_event_type', 'other_event_modality', 'other_event_polarity',
                     'other_sent_ind', 'other_start', 'other_end', 'reltype', 'sent_state'])

        return table

    def get_table(self, docs, folder):
        """

        :param docs:
        :param folder:
        :return:
        """
        # extract info from each document and concatenate them together in one big table
        problems, temp_rel = zip(*[self.extract_table(os.path.join(folder, f'{doc_id}.xml.extent'),
                                                      os.path.join(folder, f'{doc_id}.xml.tlink'),
                                                      doc_id)
                                   for doc_id in docs])

        table_problems = pd.concat(problems, ignore_index=True)
        table_temp_rel = pd.concat(temp_rel, ignore_index=True)

        return table_problems, table_temp_rel

    def return_temprel_table(self, temprel_table):
        """

        :param temprel_table:
        :return:
        """
        print("create split lists")
        split_list = self.create_split_list(pref='other')

        print("split columns in full table")
        # split columns of this full table
        table = self.split_cols(temprel_table, split_list)

        print("add attributes into table")
        # add more attributes and filter only necessary columns for last full table
        full_table = self.add_attr(table)

        print("sort table")
        # sort table
        full_table.sort_values(by='doc_id', ignore_index=True, inplace=True)

        print("convert numeric values in columns")
        # convert numeric values
        full_table[["sent_ind", "start", "end", "other_sent_ind", "other_start", "other_end"]] \
            = full_table[["sent_ind", "start", "end", "other_sent_ind", "other_start", "other_end"]].apply(
            pd.to_numeric)

        return full_table

    def return_problem_table(self, problem_table):
        """

        :param problem_table:
        :return:
        """
        print('create split lists')
        split_list = self.create_split_list()
        
        print("split columns in full problem table")
        table = self.split_cols(problem_table, split_list)
        
        print("filter event value")
        table = self.filter_out_value(table, 'event', upper=False)
        
        # last table
        full_table = table[['doc_id', 'event', 'modality', 'polarity', 'sent_ind', 'start', 'end']]
        
        print("sort table")
        # sort table
        full_table.sort_values(by='doc_id', ignore_index=True, inplace=True)
        
        # convert numeric values
        print("convert numeric values in columns")
        full_table[["sent_ind", "start", "end"]] = full_table[["sent_ind", "start", "end"]].apply(pd.to_numeric)
        
        return full_table

    def assign_context(self, df, full=True):
        """
        assign context for table problem, within and cross
        :param df:
        :param full:
        :return:
        """

        problems = df.copy(deep=True)

        problems[['left', 'right', 'left_cross', 'right_cross']] = problems.apply(
            lambda row: self.find_context(row['sent_ind'], row['start'], row['end'], row['doc_id'], full=full), axis=1,
            result_type='expand')
        return problems

    @staticmethod
    def extract_mod_pol(df):
        """

        :param df:
        :return:
        """
        problem_table = df[["event", "doc_id", "sent_ind", "start", "end", "modality", "polarity", "left", "right"]]
        problem_table["left"] = problem_table.left.apply(lambda x: ' '.join(x))
        problem_table["right"] = problem_table.right.apply(lambda x: ' '.join(x))

        # combine sent
        problem_table["sent"] = problem_table.left + ' <p> ' + problem_table.event + ' </p> ' + problem_table.right

        # save pol table
        print('Extract POLARITY table')
        problem_table_pol = problem_table[["event", "doc_id", "sent_ind", "start", "end", "polarity", "left", "right"]]
        problem_table_pol['label'] = 0
        problem_table_pol.loc[(problem_table_pol.polarity == 'POS'), 'label'] = 1
        
        # save mod table
        print('Extract MODALITY table')
        problem_table_mod = problem_table[["event", "doc_id", "sent_ind", "start", "end", "modality", "left", "right"]]
        problem_table_mod['label'] = 0
        problem_table_mod.loc[(problem_table_mod.modality == 'FACTUAL'), 'label'] = 1

        return problem_table_pol, problem_table_mod

    @staticmethod
    def insert_tags(event, start, end, context, tags):
        """

        :param event:
        :param start:
        :param end:
        :param context:
        :param tags:
        :return:
        """
        ev_s, ev_e = tags
        
        if start == end:
            assert(context[start] == event)
            context[start] = f'{ev_s} {context[start]} {ev_e}'
        else:
            tokens = event.split(' ')
            assert(context[start] == tokens[0])
            assert(' '.join(context[start:end+1]) == event)
            context[start] = f'{ev_s} {tokens[0]}'

            assert(context[end] == tokens[-1])
            context[end] = f'{tokens[-1]} {ev_e}'

        return context

    def mark_sec_event_within(self, event, other, left, right, starts, ends, tags):
        """

        :param event:
        :param other:
        :param left:
        :param right:
        :param starts:
        :param ends:
        :param tags:
        :return:
        """
        e_tags, p_tags = tags 

        start1, start2 = starts 
        end1, end2 = ends
        new_right, new_left = right.copy(), left.copy()
        
        # event on right context
        if start1 < start2:
            new_right = self.insert_tags(event=other, start=start2 - end1 - 1, end=end2 - end1 - 1,
                                         context=new_right, tags=e_tags)
            
        # event on left context
        elif start1 > start2:
            new_left = self.insert_tags(event=other, start=start2, end=end2, context=new_left, tags=e_tags)

        new_left = ' '.join(new_left)
        new_right = ' '.join(new_right)

        # get tags for problem
        pr_s, pr_e = p_tags
        
        sent = f"{new_left} {pr_s} {event} {pr_e} {new_right}"
        return sent 

    def mark_sec_event_cross(self, other, left_cross, right_cross, sent_inds, start, end, event_tags):
        """

        :param other:
        :param left_cross:
        :param right_cross:
        :param sent_inds:
        :param start:
        :param end:
        :param tags:
        :return:
        """
        sent1, sent2 = sent_inds
        left, right = left_cross.copy(), right_cross.copy()
        
        if sent1 < sent2:
            # right context
            context_right = right[sent2-sent1].copy()
            
            right[sent2-sent1] = self.insert_tags(event=other, start=start, end=end,
                                                  context=context_right, tags=event_tags)
            
        elif sent1 > sent2:
            
            # left context
            context_left = left[sent2-1].copy()
            
            left[sent2-1] = self.insert_tags(event=other, start=start, end=end, context=context_left, tags=event_tags)
            
        return left, right

    def mark_sec_event_universal(self, event, other, left, right, sent_inds, starts, ends, state, tags):
        """

        :param event:
        :param other:
        :param left:
        :param right:
        :param sent_inds:
        :param starts:
        :param ends:
        :param state:
        :param tags:
        :return:
        """
        if state == 'within':
            
            within_left, within_right = left[-1].copy(), right[0].copy()

            # find size from cross sentence that still need to be padded
            left_wsize = self.window_size - len(within_left)
            right_wsize = self.window_size - len(within_right)
            
            sent = self.mark_sec_event_within(event=event, other=other, left=within_left, right=within_right,
                                                starts=starts, ends=ends, tags=tags)

            # get tokens cross sentences
            if len(left) > 1:
                left_cross = list(chain(*left[:-1]))
                    
                text_left = ' '.join(left_cross[-left_wsize:]) if len(left_cross) > left_wsize else ' '.join(left_cross)
                
            else:
                text_left = ''
                
            if len(right) > 1:
                right_cross = list(chain(*right[1:]))
                
                text_right = ' '.join(right_cross[:right_wsize+1]) if len(right_cross) > right_wsize \
                                            else ' '.join(right_cross)
                
            else:
                text_right = ''
                
            return f'{text_left} {sent} {text_right}'

        elif state == 'cross':
            pr_s, pr_e = tags[1]
            
            cross_left, cross_right = self.mark_sec_event_cross(event=event, other=other,
                                                                left_cross=left, right_cross=right, sent_inds=sent_inds,
                                                                start=starts[1], end=ends[1], tags=tags[1])
            
            left_cross = list(chain(*cross_left))
            right_cross = list(chain(*cross_right))
            
            text_left = ' '.join(left_cross[-self.window_size:]) if len(left_cross) > self.window_size \
                                        else ' '.join(left_cross)
            text_right = ' '.join(right_cross[:self.window_size+1]) if len(right_cross) > self.window_size \
                                        else ' '.join(right_cross)
        
            return f'{text_left} {pr_s} {event} {pr_e} {text_right}'

    def extract_temprel_within(self, temprel_df, problem_df, tags):
        """

        :param temprel_df:
        :param problem_df:
        :param tags:
        :return:
        """
        # get WITHIN table
        table = temprel_df[temprel_df['sent_state'] != 'cross'].reset_index(drop=True)

        # merge with context
        table = pd.merge(left=table, right=problem_df[['event', 'doc_id', 'sent_ind', 'start', 'end', 'left', 'right']],
                            on=['event', 'doc_id', 'sent_ind', 'start', 'end'])

        # mark EVENT in context
        table['sent'] = table.apply(lambda row: self.mark_sec_event_within(event=row['event'], other=row['other_event'],
                                    left=row['left'], right=row['right'],
                                    starts=[row['start'], row['other_start']], ends=[row['end'], row['other_end']],
                                    tags=tags),
                                    axis=1, result_type='expand')

        # final table
        temprel_table = table.drop(columns=['left', 'right'])

        # assign label
        temprel_table['label'] = 0
        temprel_table.loc[(temprel_table.reltype == 'BEFORE'), 'label'] = 1
        temprel_table.loc[(temprel_table.reltype == 'AFTER'), 'label'] = 2

        return temprel_table

    def extract_temprel_universal(self, temprel_df, problem_df, tags):
        """

        :param temprel_df:
        :param problem_df:
        :param tags:
        :return:
        """
        # get UNIVERSAL table
        table = temprel_df.copy(deep=True)

        # merge with context
        table = pd.merge(left=table, right=problem_df[['event', 'doc_id', 'sent_ind', 'start', 'end',
                                                       'left_cross', 'right_cross']],
                            on=['event', 'doc_id', 'sent_ind', 'start', 'end'])

        # mark EVENT in context
        table['sent'] = table.apply(lambda row:
                                    self.mark_sec_event_universal(
                                        event=row['event'], other=row['other_event'],
                                        left=row['left_cross'], right=row['right_cross'],
                                        sent_inds=[row['sent_ind'], row['other_sent_ind']], 
                                        starts=[row['start'], row['other_start']], 
                                        ends=[row['end'], row['other_end']],
                                        state=row['sent_state'], tags=tags), 
                                    axis=1, result_type='expand')

        # final table
        temprel_table = table.drop(columns=['left_cross', 'right_cross'])
        
        # assign label
        temprel_table['label'] = 0
        temprel_table.loc[(temprel_table.reltype == 'BEFORE'), 'label'] = 1
        temprel_table.loc[(temprel_table.reltype == 'AFTER'), 'label'] = 2

        return temprel_table

    def __call__(self):
        """

        :return:
        """
        # GET TABLES temprel and problem
        temprel_raw, probs_raw = self.get_table(self.docs, self.srcfolder)

        temprel = self.return_temprel_table(temprel_raw)
        probs = self.return_problem_table(probs_raw)

        # ASSIGN CONTEXT TO PROBLEMS (within and cross)

        problems = self.assign_context(probs)

        # GET AND SAVE PROBLEMS TABLES WITH CONTEXTS mod and pol
        pol_table, mod_table = self.extract_mod_pol(problems)
    
        print('Save POLARITY table')
        self.saver.save_input_frame(pol_table, f'problems_{self.data_type}_data_2012_pol')

        print('Save MODALITY table')
        self.saver.save_input_frame(mod_table, f'problems_{self.data_type}_data_2012_mod')

        # GET AND SAVE TEMPREL TABLE WITHIN SENTENCE
        temprel_within_table = self.extract_temprel_within(temprel, problems,
                                                    tags=self.tags.get('xml', [('<e2>', '</e2>'), ('<e1>', '</e2>')]))
        self.saver.save_input_frame(temprel_within_table, f'temprel_within_{self.data_type}_2012')

        # GET AND SAVE TEMPREL TABLE CROSS SENTENCE
        temprel_universal_table = self.extract_temprel_universal(temprel, problems,
                                                        tags=self.tags.get('nonxml', [('ebs', 'ebe'), ('eas', 'eae')]))
        self.saver.save_input_frame(temprel_universal_table, f'temprel_universal_{self.data_type}_2012')

