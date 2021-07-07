# --------------------------------------------------------
# This code is modified from Jumpin2's repository.
# https://github.com/Jumpin2/HGA
# --------------------------------------------------------

"""Generate data batch."""
import os

import pandas as pd
import numpy as np
import h5py
from torch.utils.data import Dataset
from . import data_util

from util import log

import os.path
import sys

import pickle as pkl
import nltk


nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


__PATH__ = os.path.abspath(os.path.dirname(__file__))


def assert_exists(path):
    assert os.path.exists(path), 'Does not exist : {}'.format(path)


# PATHS
TYPE_TO_CSV = {
    'FrameQA': 'Train_frameqa_question.csv',
    'Count': 'Train_count_question.csv',
    'Trans': 'Train_transition_question.csv',
    'Action': 'Train_action_question.csv'
}

eos_word = '<EOS>'


class TGIFQA(Dataset):

    def __init__(
            self,
            dataset_name='train',
            q_max_length=20,
            v_max_length=80,
            max_n_videos=None,
            data_type='FrameQA',
            csv_dir=None,
            vocab_dir=None,
            feat_dir=None):
        self.csv_dir = csv_dir
        self.vocabulary_dir = vocab_dir
        self.feat_dir = feat_dir
        self.dataset_name = dataset_name
        self.q_max_length = q_max_length
        self.v_max_length = v_max_length
        self.spa_q_max_length = 10
        self.tmp_q_max_length = 7
        self.obj_max_num = 10
        self.max_n_videos = max_n_videos
        self.data_type = data_type
        self.GLOVE_EMBEDDING_SIZE = 300

        self.data_df = self.read_df_from_csvfile()

        if max_n_videos is not None:
            self.data_df = self.data_df[:max_n_videos]

        self.res_avg_file = os.path.join(self.feat_dir, "feat_r6_vsync2/res152_avgpool.hdf5")
        self.i3d_avg_file = os.path.join(self.feat_dir, "feat_r6_vsync2/tgif_i3d_hw7_perclip_avgpool.hdf5")

        self.res_avg_feat = None
        self.i3d_avg_feat = None

        self.res_roi_file = os.path.join(self.feat_dir, "feat_r6_vsync2/tgif_btup_f_obj10.hdf5")
        self.i3d_roi_file = os.path.join(self.feat_dir, "feat_r6_vsync2/tgif_i3d_roialign_hw7_perclip_avgpool_obj10.hdf5")

        self.res_roi_feat = None
        self.i3d_roi_feat = None

        self.load_word_vocabulary()
        if self.data_type == 'FrameQA':
            self.get_FrameQA_result()
        elif self.data_type == 'Count':
            self.get_Count_result()
        elif self.data_type == 'Trans':
            self.get_Trans_result()
        elif self.data_type == 'Action':
            self.get_Trans_result()

    def __getitem__(self, index):
        if self.data_type == 'FrameQA':
            return self.getitem_frameqa(index)
        elif self.data_type == 'Count':
            return self.getitem_count(index)
        elif self.data_type == 'Trans':
            return self.getitem_trans(index)
        elif self.data_type == 'Action':
            return self.getitem_trans(index)

    def __len__(self):
        if self.max_n_videos is not None:
            if self.max_n_videos <= len(self.data_df):
                return self.max_n_videos
        return len(self.data_df)

    def read_df_from_csvfile(self):
        assert self.data_type in [
            'FrameQA', 'Count', 'Trans', 'Action'
        ], 'Should choose data type '

        if self.data_type == 'FrameQA':
            train_data_path = os.path.join(
                self.csv_dir, 'Train_frameqa_question.csv')
            test_data_path = os.path.join(
                self.csv_dir, 'Test_frameqa_question.csv')
            #self.total_q = pd.DataFrame().from_csv(os.path.join(self.csv_dir,'Total_frameqa_question.csv'), sep='\t')
            self.total_q = pd.read_csv(
                os.path.join(self.csv_dir, 'Total_frameqa_question.csv'),
                sep='\t')
        elif self.data_type == 'Count':
            train_data_path = os.path.join(
                self.csv_dir, 'Train_count_question.csv')
            test_data_path = os.path.join(
                self.csv_dir, 'Test_count_question.csv')
            #self.total_q = pd.DataFrame().from_csv(os.path.join(self.csv_dir,'Total_count_question.csv'), sep='\t')
            self.total_q = pd.read_csv(
                os.path.join(self.csv_dir, 'Total_count_question.csv'),
                sep='\t')
        elif self.data_type == 'Trans':
            train_data_path = os.path.join(
                self.csv_dir, 'Train_transition_question.csv')
            test_data_path = os.path.join(
                self.csv_dir, 'Test_transition_question.csv')
            #self.total_q = pd.DataFrame().from_csv(os.path.join(self.csv_dir,'Total_transition_question.csv'), sep='\t')
            self.total_q = pd.read_csv(
                os.path.join(self.csv_dir, 'Total_transition_question.csv'),
                sep='\t')
        elif self.data_type == 'Action':
            train_data_path = os.path.join(
                self.csv_dir, 'Train_action_question.csv')
            test_data_path = os.path.join(
                self.csv_dir, 'Test_action_question.csv')
            # self.total_q = pd.DataFrame().from_csv(os.path.join(self.csv_dir,'Total_action_question.csv'), sep='\t')
            self.total_q = pd.read_csv(
                os.path.join(self.csv_dir, 'Total_action_question.csv'),
                sep='\t')

        assert_exists(train_data_path)
        assert_exists(test_data_path)

        if self.dataset_name == 'train':
            data_df = pd.read_csv(train_data_path, sep='\t')
        elif self.dataset_name == 'test':
            data_df = pd.read_csv(test_data_path, sep='\t')

        data_df = data_df.set_index('vid_id')
        data_df['row_index'] = list(
            range(1,
                  len(data_df) + 1))  # assign csv row index
        return data_df

    @property
    def n_words(self):
        ''' The dictionary size. '''
        if not hasattr(self, 'word2idx'):
            raise Exception('Dictionary not built yet!')
        return len(self.word2idx)

    def __repr__(self):
        if hasattr(self, 'word2idx'):
            return '<Dataset (%s) with %d videos and %d words>' % (
                self.dataset_name, len(self), len(self.word2idx))
        else:
            return '<Dataset (%s) with %d videos -- dictionary not built>' % (
                self.dataset_name, len(self))

    def split_sentence_into_words(self, sentence, eos=True):
        '''
        Split the given sentence (str) and enumerate the words as strs.
        Each word is normalized, i.e. lower-cased, non-alphabet characters
        like period (.) or comma (,) are stripped.
        When tokenizing, I use ``data_util.clean_str``
        '''
        try:
            words = data_util.clean_str(sentence).split()
        except:
            print(sentence)
            sys.exit()
        if eos:
            words = words + [eos_word]
        for w in words:
            if not w:
                continue
            yield w

    def build_word_vocabulary(
            self,
            all_captions_source=None,
            word_count_threshold=0,
    ):
        '''
        borrowed this implementation from @karpathy's neuraltalk.
        '''
        log.infov('Building word vocabulary (%s) ...', self.dataset_name)

        if all_captions_source is None:
            all_captions_source = self.get_all_captions()

        # enumerate all sentences to build frequency table
        word_counts = {}
        nsents = 0
        nwords = 0
        for sentence in all_captions_source:
            nsents += 1
            for w in self.split_sentence_into_words(sentence):
                word_counts[w] = word_counts.get(w, 0) + 1
                nwords += 1

        import pickle as pkl
        vocab = [
            w for w in word_counts if word_counts[w] >= word_count_threshold
        ]
        print("[%d] : [%d] word from captions" % (len(vocab), nwords))
        log.info(
            "Filtered vocab words (threshold = %d), from %d to %d",
            word_count_threshold, len(word_counts), len(vocab))

        # build index and vocabularies
        self.word2idx = {}
        self.idx2word = {}

        self.idx2word[0] = '.'
        self.idx2word[1] = 'UNK'
        self.word2idx['#START#'] = 0
        self.word2idx['UNK'] = 1
        for idx, w in enumerate(vocab, start=2):
            self.word2idx[w] = idx
            self.idx2word[idx] = w

        pkl.dump(
            self.word2idx,
            open(
                os.path.join(
                    self.vocabulary_dir,
                    'word_to_index_%s.pkl' % self.data_type), 'wb'))
        pkl.dump(
            self.idx2word,
            open(
                os.path.join(
                    self.vocabulary_dir,
                    'index_to_word_%s.pkl' % self.data_type), 'wb'))

        word_counts['.'] = nsents
        bias_init_vector = np.array(
            [
                1.0 * word_counts[w] if i > 1 else 0
                for i, w in self.idx2word.items()
            ])
        bias_init_vector /= np.sum(bias_init_vector)  # normalize to frequencies
        bias_init_vector = np.log(bias_init_vector + 1e-20)
        bias_init_vector -= np.max(
            bias_init_vector)  # shift to nice numeric range
        self.bias_init_vector = bias_init_vector

        #self.total_q = pd.DataFrame().from_csv(os.path.join(csv_dir,'Total_desc_question.csv'), sep='\t')
        answers = list(set(self.total_q['answer'].values))
        self.ans2idx = {}
        self.idx2ans = {}
        for idx, w in enumerate(answers):
            self.ans2idx[w] = idx
            self.idx2ans[idx] = w
        pkl.dump(
            self.ans2idx,
            open(
                os.path.join(
                    self.vocabulary_dir,
                    'ans_to_index_%s.pkl' % self.data_type), 'wb'))
        pkl.dump(
            self.idx2ans,
            open(
                os.path.join(
                    self.vocabulary_dir,
                    'index_to_ans_%s.pkl' % self.data_type), 'wb'))

        # Make glove embedding.
        #import spacy
        #nlp = spacy.load('en', vectors='en_glove_cc_300_1m_vectors')

        with open(self.vocabulary_dir + '/glove.42B.300d.txt', 'rb') as f:
            lines = f.readlines()
        dict = {}

        for ix, line in enumerate(lines):
            if ix % 1000 == 0:
                print((ix, len(lines)))
            segs = line.split()
            wd = segs[0]
            dict[wd] = segs[1:]

        #max_length = len(vocab)
        max_length = len(self.idx2word)
        GLOVE_EMBEDDING_SIZE = 300

        glove_matrix = np.random.normal(size=[max_length, GLOVE_EMBEDDING_SIZE])
        not_in_vocab = 0
        for i in range(len(vocab)):
            if i % 1000 == 0:
                print((i, len(vocab)))
            w = bytes(vocab[i], encoding='utf-8')
            #w_embed = nlp(u'%s' % w).vector
            if w in dict:
                w_embed = np.array(dict[w])
            else:
                not_in_vocab += 1
                w_embed = np.random.normal(size=(GLOVE_EMBEDDING_SIZE))
            #glove_matrix[i,:] = w_embed
            glove_matrix[i + 2, :] = w_embed  # two placeholder words '.','UNK'

        print("[%d/%d] word is not saved" % (not_in_vocab, len(vocab)))

        vocab = pkl.dump(
            glove_matrix,
            open(
                os.path.join(
                    self.vocabulary_dir,
                    'vocab_embedding_%s.pkl' % self.data_type), 'wb'))
        self.word_matrix = glove_matrix

    def load_word_vocabulary(self):

        word_matrix_path = os.path.join(
            self.vocabulary_dir, 'vocab_embedding_%s.pkl' % self.data_type)

        word2idx_path = os.path.join(
            self.vocabulary_dir, 'word_to_index_%s.pkl' % self.data_type)
        idx2word_path = os.path.join(
            self.vocabulary_dir, 'index_to_word_%s.pkl' % self.data_type)
        ans2idx_path = os.path.join(
            self.vocabulary_dir, 'ans_to_index_%s.pkl' % self.data_type)
        idx2ans_path = os.path.join(
            self.vocabulary_dir, 'index_to_ans_%s.pkl' % self.data_type)

        if not (os.path.exists(word_matrix_path) and
                os.path.exists(word2idx_path) and
                os.path.exists(idx2word_path) and
                os.path.exists(ans2idx_path) and os.path.exists(idx2ans_path)):
            self.build_word_vocabulary()

        # ndarray
        with open(word_matrix_path, 'rb') as f:
            self.word_matrix = pkl.load(f)
        log.info("Load word_matrix from pkl file : %s", word_matrix_path)

        with open(word2idx_path, 'rb') as f:
            self.word2idx = pkl.load(f)
        log.info("Load word2idx from pkl file : %s", word2idx_path)

        with open(idx2word_path, 'rb') as f:
            self.idx2word = pkl.load(f)
        log.info("Load idx2word from pkl file : %s", idx2word_path)

        with open(ans2idx_path, 'rb') as f:
            self.ans2idx = pkl.load(f)
        log.info("Load answer2idx from pkl file : %s", ans2idx_path)

        with open(idx2ans_path, 'rb') as f:
            self.idx2ans = pkl.load(f)
        log.info("Load idx2answers from pkl file : %s", idx2ans_path)

    def share_word_vocabulary_from(self, dataset):
        assert hasattr(dataset, 'idx2word') and hasattr(
            dataset, 'word2idx'
        ), 'The dataset instance should have idx2word and word2idx'
        assert (
            isinstance(dataset.idx2word, dict) or
            isinstance(dataset.idx2word, list)
        ) and isinstance(
            dataset.word2idx, dict
        ), 'The dataset instance should have idx2word and word2idx (as dict)'

        if hasattr(self, 'word2idx'):
            log.warn(
                "Overriding %s' word vocabulary from %s ...", self, dataset)

        self.idx2word = dataset.idx2word
        self.word2idx = dataset.word2idx
        self.ans2idx = dataset.ans2idx
        self.idx2ans = dataset.idx2ans
        if hasattr(dataset, 'word_matrix'):
            self.word_matrix = dataset.word_matrix

    def get_all_captions(self):
        '''
        Iterate caption strings associated in the vid/gifs.
        '''
        #qa_data_df = pd.DataFrame().from_csv(os.path.join(self.csv_dir, TYPE_TO_CSV[self.data_type]), sep='\t')
        qa_data_df = pd.read_csv(
            os.path.join(self.csv_dir, TYPE_TO_CSV[self.data_type]), sep='\t')

        all_sents = []
        for row in qa_data_df.iterrows():
            all_sents.extend(self.get_captions(row))
        self.data_type
        return all_sents

    def get_captions(self, row):
        if self.data_type == 'FrameQA':
            columns = ['description', 'question', 'answer']
        elif self.data_type == 'Count':
            columns = ['question']
        elif self.data_type == 'Trans':
            columns = ['question', 'a1', 'a2', 'a3', 'a4', 'a5']
        elif self.data_type == 'Action':
            columns = ['question', 'a1', 'a2', 'a3', 'a4', 'a5']

        sents = [row[1][col] for col in columns if not pd.isnull(row[1][col])]
        return sents

    def get_video_feature(self, key): # key : gif_name
        if self.res_avg_feat is None:
            self.res_avg_feat = h5py.File(self.res_avg_file, 'r')
        if self.i3d_avg_feat is None:
            self.i3d_avg_feat = h5py.File(self.i3d_avg_file, 'r')
        if self.res_roi_feat is None:
            self.res_roi_feat = h5py.File(self.res_roi_file, 'r')
        if self.i3d_roi_feat is None:
            self.i3d_roi_feat = h5py.File(self.i3d_roi_file, 'r')

        video_id = str(key)

        try:
            res_avg_feat = np.array(self.res_avg_feat[video_id])  # T, d
            i3d_avg_feat = np.array(self.i3d_avg_feat[video_id])  # T, d
            res_roi_feat = np.array(self.res_roi_feat['image_features'][video_id])  # T, 5, d
            roi_bbox_feat = np.array(self.res_roi_feat['spatial_features'][video_id])  # T, 5, 6
            i3d_roi_feat = np.array(self.i3d_roi_feat[video_id])  # T, 5, d
        except KeyError: # no img
            print('no image', key)
            res_avg_feat = np.zeros((1, 2048))
            i3d_avg_feat = np.zeros((1, 2048))
            res_roi_feat = np.zeros((1, self.obj_max_num, 2048))
            roi_bbox_feat = np.zeros((1, self.obj_max_num, 6))
            i3d_roi_feat = np.zeros((1, self.obj_max_num, 2048))

        return res_avg_feat, i3d_avg_feat, res_roi_feat, roi_bbox_feat, i3d_roi_feat

    def convert_sentence_to_matrix(self, task, question, answer=None, eos=True):
        '''
        Convert the given sentence into word indices and masks.
        WARNING: Unknown words (not in vocabulary) are revmoed.

        Args:
            sentence: A str for unnormalized sentence, containing T words

        Returns:
            sentence_word_indices : list of (at most) length T,
                each being a word index
        '''
        if answer is None:
            sentence = question
        else:
            sentence = question + ' ' + answer

        token = self.split_sentence_into_words(sentence, eos=False)
        token = [t for t in token]

        sent2indices = [
            self.word2idx[w] if w in self.word2idx else 1
            for w in token
        ]
        T = len(sent2indices)
        length = min(T, self.q_max_length)

        return sent2indices[:length]

    def get_question(self, key):
        '''
        Return question index for given key.
        '''
        question = self.data_df.loc[key, ['question', 'description']].values
        if len(list(question.shape)) > 1:
            question = question[0]
        # A question string
        question = question[0]

        return self.convert_sentence_to_matrix(self.data_type, question, eos=False)

    def get_answer(self, key):
        answer = self.data_df.loc[key, ['answer', 'type']].values

        if len(list(answer.shape)) > 1:
            answer = answer[0]

        anstype = answer[1]
        answer = answer[0]

        return answer, anstype

    def get_FrameQA_result(self):
        self.padded_all_ques = []
        self.answers = []
        self.answer_type = []
        self.all_ques_lengths = []

        self.keys = self.data_df['key'].values.astype(np.int64)

        for index, row in self.data_df.iterrows():
            # ====== Question ======
            all_ques = self.get_question(index)
            all_ques_pad = data_util.question_pad(all_ques, self.q_max_length)
            self.padded_all_ques.append(all_ques_pad)

            all_ques_length = min(self.q_max_length, len(all_ques))
            self.all_ques_lengths.append(all_ques_length)

            answer, answer_type = self.get_answer(index)
            if str(answer) in self.ans2idx:
                answer = self.ans2idx[answer]
            else:
                # unknown token, check later
                answer = 1
            self.answers.append(np.array(answer, dtype=np.int64))
            self.answer_type.append(np.array(answer_type, dtype=np.int64))

    def getitem_frameqa(self, index):
        key = self.keys[index]

        res_avg_feat, i3d_avg_feat, res_roi_feat, roi_bbox_feat, i3d_roi_feat = self.get_video_feature(key)

        res_avg_pad = data_util.video_pad(res_avg_feat, self.v_max_length).astype(np.float32)
        i3d_avg_pad = data_util.video_pad(i3d_avg_feat, self.v_max_length).astype(np.float32)

        res_roi_pad = data_util.video_3d_pad(res_roi_feat, self.obj_max_num,
                                         self.v_max_length).astype(np.float32)
        bbox_pad = data_util.video_3d_pad(roi_bbox_feat, self.obj_max_num,
                                          self.v_max_length).astype(np.float32)
        i3d_roi_pad = data_util.video_3d_pad(i3d_roi_feat, self.obj_max_num,
                                      self.v_max_length).astype(np.float32)
        video_length = min(self.v_max_length, res_avg_feat.shape[0])

        return res_avg_pad, i3d_avg_pad, res_roi_pad, bbox_pad, i3d_roi_pad, video_length, \
               self.padded_all_ques[index], self.all_ques_lengths[index], self.answers[index], self.answer_type[index]

    def get_Count_question(self, key):
        '''
        Return question string for given key.
        '''
        question = self.data_df.loc[key, 'question']
        return self.convert_sentence_to_matrix(self.data_type, question, eos=False)

    def get_Count_result(self):
        self.padded_all_ques = []
        self.answers = []
        self.all_ques_lengths = []

        self.keys = self.data_df['key'].values.astype(np.int64)

        for index, row in self.data_df.iterrows():
            # ====== Question ======
            all_ques = self.get_Count_question(index)
            all_ques_pad = data_util.question_pad(all_ques, self.q_max_length)

            self.padded_all_ques.append(all_ques_pad)

            all_ques_length = min(self.q_max_length, len(all_ques))
            self.all_ques_lengths.append(all_ques_length)

            # force answer not equal to 0
            answer = max(row['answer'], 1)
            self.answers.append(np.array(answer, dtype=np.float32))

    def getitem_count(self, index):
        key = self.keys[index]

        res_avg_feat, i3d_avg_feat, res_roi_feat, roi_bbox_feat, i3d_roi_feat = self.get_video_feature(key)

        res_avg_pad = data_util.video_pad(res_avg_feat, self.v_max_length).astype(np.float32)
        i3d_avg_pad = data_util.video_pad(i3d_avg_feat, self.v_max_length).astype(np.float32)

        res_roi_pad = data_util.video_3d_pad(res_roi_feat, self.obj_max_num,
                                             self.v_max_length).astype(np.float32)
        bbox_pad = data_util.video_3d_pad(roi_bbox_feat, self.obj_max_num,
                                          self.v_max_length).astype(np.float32)
        i3d_roi_pad = data_util.video_3d_pad(i3d_roi_feat, self.obj_max_num,
                                             self.v_max_length).astype(np.float32)
        video_length = min(self.v_max_length, res_avg_feat.shape[0])

        return res_avg_pad, i3d_avg_pad, res_roi_pad, bbox_pad, i3d_roi_pad, video_length, \
               self.padded_all_ques[index], self.all_ques_lengths[index], self.answers[index]

    def get_Trans_matrix(self, candidates, q_max_length, is_left=True):
        candidates_matrix = np.zeros([5, q_max_length], dtype=np.int64)
        for k in range(5):
            sentence = candidates[k]
            if is_left:
                candidates_matrix[k, :len(sentence)] = sentence
            else:
                candidates_matrix[k, -len(sentence):] = sentence
        return candidates_matrix

    def get_Trans_result(self):
        self.padded_candidates = []
        self.answers = self.data_df['answer'].values.astype(np.int64)
        self.row_index = self.data_df['row_index'].values.astype(np.int64)
        self.candidate_lengths = []

        self.keys = self.data_df['key'].values.astype(np.int64)

        for index, row in self.data_df.iterrows():
            a1 = row['a1'].strip()
            a2 = row['a2'].strip()
            a3 = row['a3'].strip()
            a4 = row['a4'].strip()
            a5 = row['a5'].strip()
            candidates = [a1, a2, a3, a4, a5]
            raw_question = row['question'].strip()
            indexed_candidates = []

            for x in candidates:
                all_cand = self.convert_sentence_to_matrix(self.data_type, raw_question, x, eos=False)
                indexed_candidates.append(all_cand)

            cand_lens = []
            for i in range(len(indexed_candidates)):
                vl = min(self.q_max_length, len(indexed_candidates[i]))
                cand_lens.append(vl)
            self.candidate_lengths.append(cand_lens)

            # (5, self.q_max_length)
            candidates_pad = self.get_Trans_matrix(indexed_candidates, self.q_max_length)
            self.padded_candidates.append(candidates_pad)

    def getitem_trans(self, index):
        key = self.keys[index]

        res_avg_feat, i3d_avg_feat, res_roi_feat, roi_bbox_feat, i3d_roi_feat = self.get_video_feature(key)

        res_avg_pad = data_util.video_pad(res_avg_feat, self.v_max_length).astype(np.float32)
        i3d_avg_pad = data_util.video_pad(i3d_avg_feat, self.v_max_length).astype(np.float32)

        res_roi_pad = data_util.video_3d_pad(res_roi_feat, self.obj_max_num,
                                             self.v_max_length).astype(np.float32)
        bbox_pad = data_util.video_3d_pad(roi_bbox_feat, self.obj_max_num,
                                          self.v_max_length).astype(np.float32)
        i3d_roi_pad = data_util.video_3d_pad(i3d_roi_feat, self.obj_max_num,
                                             self.v_max_length).astype(np.float32)
        video_length = min(self.v_max_length, res_avg_feat.shape[0])

        self.padded_candidates = np.asarray(self.padded_candidates).astype(
            np.int64)
        self.candidate_lengths = np.asarray(self.candidate_lengths).astype(
            np.int64)

        return res_avg_pad, i3d_avg_pad, res_roi_pad, bbox_pad, i3d_roi_pad, video_length, \
               self.padded_candidates[index], self.candidate_lengths[index], self.answers[index], self.row_index[index]
