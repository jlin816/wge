import abc
import os
import string
import torch

import numpy as np
import logging
from transformers import BertModel, BertTokenizerFast
from torch.nn.utils.rnn import pad_sequence

from gtd.chrono import verboserate
from gtd.ml.torch.seq_batch import SequenceBatch, SequenceBatchElement
from gtd.ml.torch.source_encoder import BidirectionalSourceEncoder
from gtd.ml.torch.token_embedder import TokenEmbedder
from gtd.ml.vocab import SimpleEmbeddings, SimpleVocab
from gtd.utils import random_seed
from torch.nn import Module, LSTMCell, LSTM
from gtd.ml.torch.utils import GPUVariable
from wge import data
from wge.cache import Cache
from wge.miniwob import special_words
from wge.vocab import LazyInitVocab


# TODO: A lot of these defined embeddings, should have a randomly initialized
# embeddings class, that takes a vocab and initializes the embed matrix
class BoolEmbeddings(SimpleEmbeddings):
    """Embed boolean values"""
    def __init__(self, embed_dim):
        bool_vocab = SimpleVocab([True, False])
        embed_matrix = np.random.uniform(
                -np.sqrt(3. / embed_dim), np.sqrt(3. / embed_dim),
                size=(len(bool_vocab), embed_dim)).astype(np.float32)
        super(BoolEmbeddings, self).__init__(embed_matrix, bool_vocab)


class Embedder(Module):
    """An Embedder, which takes objects and returns embeddings in its forward
    method.
    """
    @abc.abstractproperty
    def embed_dim(self):
        raise NotImplementedError()


class LazyInitEmbeddings(SimpleEmbeddings):
    """Can embed new words on the fly. If the current size is less than
    max_size, then it assigns the new word a new randomly initialized embed,
    otherwise it assigns it UNK.
    """
    def __init__(self, embed_dim, max_size):
        """Create embeddings object.

        Args:
            embed_dim (int)
            max_size (int): maximum size of vocab
        """
        # Initialize vocab empty
        lazy_vocab = LazyInitVocab([], max_size)
        embed_matrix = np.random.uniform(
                -np.sqrt(3. / embed_dim), np.sqrt(3. / embed_dim),
                size=(len(lazy_vocab), embed_dim)).astype(np.float32)
        super(LazyInitEmbeddings, self).__init__(embed_matrix, lazy_vocab)


class CachedEmbedder(Embedder):
    """Embedding object that uses caching."""
    def __init__(self):
        super(CachedEmbedder, self).__init__()
        self._cache = Cache()

    @abc.abstractmethod
    def clear_cache(self):
        """Clears cache and all submodule caches."""
        raise NotImplementedError()


class GloveEmbeddings(SimpleEmbeddings):
    def __init__(self, vocab_size=400000):
        """Load GloveEmbeddings.

        Args:
            word_vocab_size (int): max # of words in the vocab. If not
                specified, uses all available GloVe vectors.
        """
        embed_dim = 100
        if vocab_size < 5000:
            raise ValueError('Need to at least use 5000 words.')

        glove_path = os.path.join(data.workspace.glove, 'glove.6B.100d.txt')
        download_path = 'http://nlp.stanford.edu/data/glove.6B.zip'
        if not os.path.exists(glove_path):
            raise RuntimeError('Missing file: {}. Download it here: {}'
                               .format(glove_path, download_path))

        # embeddings for special words
        words = list(UtteranceVocab.SPECIAL_TOKENS)
        num_special = len(words)

        # zeros are just placeholders for now
        embeds = [np.zeros(embed_dim, dtype=np.float32) for _ in words]

        with open(glove_path, 'r') as f:
            lines = verboserate(f, desc='Loading GloVe embeddings',
                                total=vocab_size, initial=num_special)
            for i, line in enumerate(lines, start=num_special):
                if i == vocab_size:
                    break
                tokens = line.split()
                word = tokens[0]
                embed = np.array([float(tok) for tok in tokens[1:]])
                words.append(word)
                embeds.append(embed)

        # TODO: Vocab size is slightly off now
        # TODO: This is no longer non-MiniWoB specific
        extra_words = set(special_words.NAMES) - set(words)
        words += extra_words
        embeds += [np.zeros(embed_dim, dtype=np.float32) for _ in extra_words]

        vocab = UtteranceVocab(words)
        embed_matrix = np.stack(embeds).astype(np.float32)

        # Embeddings for PAD / UNK
        special_embeds = emulate_distribution(
                (num_special, embed_dim), embed_matrix[:5000, :], seed=2)
        embed_matrix[:num_special, :] = special_embeds

        # Embeddings for extra words
        extra_word_embeds = emulate_distribution(
                (len(extra_words), embed_dim), embed_matrix[:5000, :], seed=2)
        embed_matrix[vocab_size:, :] = extra_word_embeds
        assert embed_matrix.shape[1] == 100

        super(GloveEmbeddings, self).__init__(embed_matrix, vocab)


def emulate_distribution(shape, target_samples, seed=None):
    m = np.mean(target_samples)
    s = np.std(target_samples)

    with random_seed(seed):
        samples = np.random.normal(m, s, size=shape)

    return samples


class UtteranceVocab(SimpleVocab):
    """Vocab for input utterances.

    IMPORTANT NOTE: UtteranceVocab is blind to casing! All words are
    converted to lower-case.

    An UtteranceVocab is required to have the following special tokens: UNK,
    PAD. See class attributes for more info.
    """
    UNK = "<unk>"
    PAD = "<pad>"
    SPECIAL_TOKENS = (UNK, PAD)

    def __init__(self, tokens):
        tokens = [t.lower() for t in tokens]
        super(UtteranceVocab, self).__init__(tokens)

        # check that all special tokens present
        for special in self.SPECIAL_TOKENS:
            if special not in self._word2index:
                raise ValueError(('All special tokens must be present '
                                  'in tokens. Missing {}').format(special))

    def word2index(self, w):
        """Map a word to an integer.

        If the word is not known to the vocab, return the index for UNK.
        """
        sup = super(UtteranceVocab, self)
        try:
            return sup.word2index(w.lower())
        except KeyError:
            logging.info("%s embedded as UNK", w)
            return sup.word2index(self.UNK)

class UtteranceEmbedder(CachedEmbedder):

    @classmethod
    def from_config(cls, config):
        """Constructs the appropriate UtteranceEmbedder from a config.

        Args:
            config (Config)

        Returns:
            UtteranceEmbedder
        """
        if config.type == "glove":
            glove_embeddings = GloveEmbeddings(config.vocab_size)
            token_embedder = TokenEmbedder(glove_embeddings, trainable=False)
            utterance_embedder = LSTMUtteranceEmbedder(token_embedder, config.lstm_dim)
        elif config.type == "bert":
            utterance_embedder = BERTUtteranceEmbedder(freeze_bert=config.freeze_bert) 
        else:
            raise ValueError(
                "{} not a supported type of utterance embedder".format(
                    config.type))
        return utterance_embedder

    def clear_cache(self):
        # Keep empty tuple cached, for SequenceBatch
        self._cache.clear()
        self._cache.cache(
            [tuple()], [
                (GPUVariable(torch.zeros(self._embed_dim)),
                 SequenceBatchElement(
                     GPUVariable(torch.zeros(1, self._embed_dim)),
                     GPUVariable(torch.zeros(1)))
                 )])

    @property
    def embed_dim(self):
        return self._embed_dim

class LSTMUtteranceEmbedder(UtteranceEmbedder):
    """Takes a string, embeds the tokens using the token_embedder, and passes
    the embeddings through a biLSTM padded / masked up to sequence_length.
    Returns the concatenation of the two front and end hidden states.

    Args:
        token_embedder (TokenEmbedder): used to embed each token
        lstm_dim (int): output dim of the lstm
    """
    def __init__(self, token_embedder, lstm_dim):
        super(UtteranceEmbedder, self).__init__()

        self._token_embedder = token_embedder

        self._bilstm = BidirectionalSourceEncoder(
               token_embedder.embed_dim, lstm_dim, LSTMCell)
        self._embed_dim = lstm_dim

        self._cache = Cache()  # tuple[unicode] --> Variable[FloatTensor]
        self.clear_cache()

    def forward(self, utterance):
        """Embeds a batch of utterances.

        Args:
            utterance (list[list[unicode]]): list[unicode] is a list of tokens
            forming a sentence. list[list[unicode]] is batch of sentences.

        Returns:
            Variable[FloatTensor]: batch x lstm_dim
                (concatenated first and last hidden states)
            list[SequenceBatchElement]: list of length batch, where each
                element's values is seq_len x embed_dim and mask is seq_len,
                representing the hidden states of each token.
        """
        # Make keys hashable
        utterance = [tuple(utt) for utt in utterance]

        uncached_utterances = self._cache.uncached_keys(utterance)

        # Cache the uncached utterances
        if len(uncached_utterances) > 0:
            token_indices = SequenceBatch.from_sequences(
                    uncached_utterances, self._token_embedder.vocab)
            # batch x seq_len x token_embed_dim
            token_embeds = self._token_embedder.embed_seq_batch(token_indices)

            bi_hidden_states = self._bilstm(token_embeds.split())
            final_states = torch.cat(bi_hidden_states.final_states, 1)

            # Store the combined states in batch x stuff order for caching.
            combined_states = bi_hidden_states.combined_states
            # batch x seq_len x embed_dim
            combined_values = torch.stack(
                    [state.values for state in combined_states], 1)
            # batch x seq_len
            combined_masks = torch.stack(
                    [state.mask for state in combined_states], 1)
            assert len(combined_values) == len(combined_masks)
            combined_states_by_batch = [SequenceBatchElement(
                value, mask) for value, mask in zip(
                    combined_values, combined_masks)]

            assert len(final_states) == len(combined_states_by_batch)
            self._cache.cache(
                uncached_utterances,
                list(zip(final_states, combined_states_by_batch)))

        final_states, combined_states = list(zip(*self._cache.get(utterance)))
        return torch.stack(final_states, 0), combined_states

class BERTUtteranceEmbedder(UtteranceEmbedder):

    def __init__(self, freeze_bert=True):
        super().__init__()
        self._cache = Cache() # tuple[unicode] -> Variable[FloatTensor]
        self._bert = BertModel.from_pretrained("bert-base-uncased")
        self.freeze_bert = freeze_bert
        if self.freeze_bert:
            for param in self._bert.parameters():
                param.requires_grad = False

        self._bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self._embed_dim = 768
        self.clear_cache()

    def _convert_pad_tokens_to_bert(self, utt):
        return utt.replace("<pad>", "[PAD]")

    def forward(self, utterance):
        """
        Args:
            utterance (list[list[unicode]]): list[unicode] is a list of tokens
            forming a sentence. list[list[unicode]] is batch of sentences.
        
        Returns:
            utt_embedding: (batch, hidden_dim)
            per_token_utt_embedding: list[SequenceBatchElement] *Not implemented*
        """
        # Make keys hashable
        utterance = [tuple(utt) for utt in utterance]

        uncached_utterances = self._cache.uncached_keys(utterance)

        # Cache the uncached utterances
        if len(uncached_utterances) > 0:
            utts = [self._convert_pad_tokens_to_bert(" ".join(utt)) for utt in uncached_utterances]
            utts = self._bert_tokenizer(utts, return_tensors="pt", padding=True).to("cuda")
            utts_ids, utts_mask = utts["input_ids"], utts["attention_mask"]
            final_states = self._bert(input_ids=utts_ids,
                    attention_mask=utts_mask).pooler_output
            self._cache.cache(
                uncached_utterances,
                list(zip(final_states, [None] * len(final_states))))

        final_states, combined_states = list(zip(*self._cache.get(utterance)))
        return torch.stack(final_states, 0), combined_states
