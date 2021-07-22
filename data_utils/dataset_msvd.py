# --------------------------------------------------------
# This code is modified from Jumpin2's repository.
# https://github.com/Jumpin2/HGA
# --------------------------------------------------------

"""Generate data batch."""
import os
import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pandas as pd
import numpy as np
import h5py
from torch.utils.data import Dataset
from data_utils import data_util

from util import log

import pickle as pkl
import nltk


nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


__PATH__ = os.path.abspath(os.path.dirname(__file__))


def assert_exists(path):
    assert os.path.exists(path), 'Does not exist : {}'.format(path)

eos_word = '<EOS>'


class MSVDQA(Dataset):

    def __init__(
            self,
            dataset_name='train',
            q_max_length=20,
            v_max_length=80,
            max_n_videos=None,
            csv_dir=None,
            vocab_dir=None,
            feat_dir=None):
        self.csv_dir = csv_dir
        self.vocabulary_dir = vocab_dir
        self.feat_dir = feat_dir
        self.dataset_name = dataset_name
        self.q_max_length = q_max_length
        self.v_max_length = v_max_length
        self.obj_max_num = 10
        self.max_n_videos = max_n_videos
        self.GLOVE_EMBEDDING_SIZE = 300

        self.data_df = self.read_df_from_json()

        if max_n_videos is not None:
            self.data_df = self.data_df[:max_n_videos]

        self.res_avg_file = os.path.join(self.feat_dir, "vfeat/msvd_res152_avgpool.hdf5")
        self.i3d_avg_file = os.path.join(self.feat_dir, "vfeat/msvd_i3d_avgpool_perclip.hdf5")

        self.res_avg_feat = None
        self.i3d_avg_feat = None

        self.res_roi_file = os.path.join(self.feat_dir, "vfeat/msvd_btup_f_obj10.hdf5")
        self.i3d_roi_file = os.path.join(self.feat_dir, "vfeat/msvd_i3d_roialign_perclip_obj10.hdf5")

        self.res_roi_feat = None
        self.i3d_roi_feat = None

        self.load_word_vocabulary()
        self.get_result()

    def __getitem__(self, index):
        return self.get_item(index)

    def __len__(self):
        if self.max_n_videos is not None:
            if self.max_n_videos <= len(self.data_df):
                return self.max_n_videos
        return len(self.data_df)

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

    def read_df_from_json(self):
        if self.dataset_name == 'train':
            data_path = '%s/train_qa.json'%self.csv_dir
        elif self.dataset_name == 'val':
            data_path = '%s/val_qa.json'%self.csv_dir
        elif self.dataset_name == 'test':
            data_path = '%s/test_qa.json'%self.csv_dir

        with open(data_path, 'r') as f:
            data_df = pd.read_json(f)

        return data_df

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

    def create_answerset(self, ans_df):
        """Generate 1000 answer set from train_qa.json.
        Args:
            trainqa_path: path to train_qa.json.
            answerset_path: generate answer set of mc_qa
        """
        answer_freq = ans_df['answer'].value_counts()
        answer_freq = list(answer_freq.iloc[0:1000].keys())
        return answer_freq

    def build_word_vocabulary(
            self,
            all_sen=None,
            ans_df=None,
            word_count_threshold=0,
    ):
        '''
        borrowed this implementation from @karpathy's neuraltalk.
        '''
        log.infov('Building word vocabulary (%s) ...', self.dataset_name)

        if all_sen is None or ans_df is None:
            all_sen, ans_df = self.get_all_captions()
        all_captions_source = all_sen

        # enumerate all sentences to build frequency table
        word_counts = {}
        nsents = 0
        nwords = 0
        for sentence in all_captions_source:
            nsents += 1
            for w in self.split_sentence_into_words(sentence):
                word_counts[w] = word_counts.get(w, 0) + 1
                nwords += 1

        vocab = [
            w for w in word_counts if word_counts[w] >= word_count_threshold
        ]
        print("Filtered vocab words (threshold = %d), from %d to %d" % (word_count_threshold, len(word_counts), len(vocab)))
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
            open(os.path.join(self.vocabulary_dir, 'word_to_index.pkl'), 'wb')
        )
        pkl.dump(
            self.idx2word,
            open(os.path.join(self.vocabulary_dir, 'index_to_word.pkl'), 'wb')
        )

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
        answers = self.create_answerset(ans_df)
        print("all answer num : %d"%len(answers))
        self.ans2idx = {}
        self.idx2ans = {}
        for idx, w in enumerate(answers):
            self.ans2idx[w] = idx
            self.idx2ans[idx] = w
        pkl.dump(
            self.ans2idx,
            open(os.path.join(self.vocabulary_dir, 'ans_to_index.pkl'), 'wb')
        )
        pkl.dump(
            self.idx2ans,
            open(os.path.join(self.vocabulary_dir, 'index_to_ans.pkl'), 'wb')
        )

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
            open(os.path.join(self.vocabulary_dir, 'vocab_embedding.pkl'), 'wb')
        )
        self.word_matrix = glove_matrix

    def load_word_vocabulary(self):

        word_matrix_path = os.path.join(
            self.vocabulary_dir, 'vocab_embedding.pkl')

        word2idx_path = os.path.join(
            self.vocabulary_dir, 'word_to_index.pkl')
        idx2word_path = os.path.join(
            self.vocabulary_dir, 'index_to_word.pkl')
        ans2idx_path = os.path.join(
            self.vocabulary_dir, 'ans_to_index.pkl')
        idx2ans_path = os.path.join(
            self.vocabulary_dir, 'index_to_ans.pkl')

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

    def get_all_captions(self):
        '''
        Iterate caption strings associated in the vid/gifs.
        '''
        data_path = '%s/train_qa.json'%self.csv_dir
        with open(data_path, 'r') as f:
            data_df = pd.read_json(f)

        all_sents = list(data_df['question'])
        return all_sents, data_df

    def get_video_feature(self, key): # key : gif_name
        if self.res_avg_feat is None:
            self.res_avg_feat = h5py.File(self.res_avg_file, 'r')
        if self.i3d_avg_feat is None:
            self.i3d_avg_feat = h5py.File(self.i3d_avg_file, 'r')
        if self.res_roi_feat is None:
            self.res_roi_feat = h5py.File(self.res_roi_file, 'r')
        if self.i3d_roi_feat is None:
            self.i3d_roi_feat = h5py.File(self.i3d_roi_file, 'r')

        video_id = 'vid' + str(key)

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

    def convert_sentence_to_matrix(self, sentence, eos=True):
        '''
        Convert the given sentence into word indices and masks.
        WARNING: Unknown words (not in vocabulary) are revmoed.

        Args:
            sentence: A str for unnormalized sentence, containing T words

        Returns:
            sentence_word_indices : list of (at most) length T,
                each being a word index
        '''
        sent2indices = [
            self.word2idx[w] if w in self.word2idx else 1
            for w in sentence
        ]  # 1 is UNK, unknown
        T = len(sent2indices)
        length = min(T, self.q_max_length)
        return sent2indices[:length]

    def get_question(self, key):
        '''
        Return question index for given key.
        '''
        question = self.data_df.loc[key, ['question']].values
        question = question[0]

        question = self.split_sentence_into_words(question, eos=False)
        q_refine = []
        for w in question:
            q_refine.append(w)
        q_refine = self.convert_sentence_to_matrix(q_refine)
        return q_refine

    def get_answer(self, key):
        answer = self.data_df.loc[key, ['answer']].values
        answer = answer[0]

        return answer

    def get_result(self):
        self.padded_all_ques = []
        self.answers = []

        self.all_ques_lengths = []
        self.keys = self.data_df['video_id'].values.astype(np.int64)

        for index, row in self.data_df.iterrows():
            # ====== Question ======
            all_ques = self.get_question(index)
            all_ques_pad = data_util.question_pad(all_ques, self.q_max_length)
            self.padded_all_ques.append(all_ques_pad)
            all_ques_length = min(self.q_max_length, len(all_ques))
            self.all_ques_lengths.append(all_ques_length)

            answer = self.get_answer(index)
            if str(answer) in self.ans2idx:
                answer = self.ans2idx[answer]
            else:
                # unknown token, check later
                answer = 1
            self.answers.append(np.array(answer, dtype=np.int64))

    def get_item(self, index):
        key = self.keys[index]

        res_avg_feat, i3d_avg_feat, res_roi_feat, roi_bbox_feat, i3d_roi_feat \
            = self.get_video_feature(key)

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
