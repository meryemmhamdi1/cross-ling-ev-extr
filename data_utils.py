try:
    from pycorenlp import StanfordCoreNLP
except:
    pass
import numpy as np
import json
import os

UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"
CUTOFF = 500


def get_iso_lang_abbreviation():
    iso_lang_dict = {}
    lang_iso_dict = {}
    with open("iso_lang_abbr.txt") as file:
        lines = file.read().splitlines()
        for line in lines:
            lang_iso_dict.update({line.split(":")[0]:line.split(":")[1]})
            iso_lang_dict.update({line.split(":")[1]:line.split(":")[0]})
    return iso_lang_dict, lang_iso_dict


def pretty_str(a):
    a = a.upper()
    if a == 'O':
        return a
    elif a[1] == '-':
        return a[:2] + "|".join(a[2:].split("-")).replace(":", "||")
    else:
        return "|".join(a.split("-")).replace(":", "||")


def generateTriggerLabelList(triggerJsonList, length):
    triggerLabel = ["O" for _ in range(length)]

    def assignTriggerLabel(index, label):
        if index >= CUTOFF:
            return
        triggerLabel[index] = pretty_str(label)

    for eventJson in triggerJsonList:
        triggerJson = eventJson["trigger"]
        start = triggerJson["start"]
        end = triggerJson["end"]
        etype = eventJson["event_type"]
        assignTriggerLabel(start, "B-" + etype)
        for i in range(start + 1, end):
            assignTriggerLabel(i, "I-" + etype)
    return triggerLabel


def generateGoldenEvents(eventsJson):
    '''

    {
        (2, 3, "event_type_str") --> [(1, 2, "role_type_str"), ...]
        ...
    }

    '''
    golden_dict = {}
    for eventJson in eventsJson:
        triggerJson = eventJson["trigger"]
        if triggerJson["start"] >= CUTOFF:
            continue
        key = (triggerJson["start"], min(triggerJson["end"], CUTOFF), pretty_str(eventJson["event_type"]))
        values = []
        for argumentJson in eventJson["arguments"]:
            if argumentJson["start"] >= CUTOFF:
                continue
            value = (argumentJson["start"], min(argumentJson["end"], CUTOFF), pretty_str(argumentJson["role"]))
            values.append(value)
        golden_dict[key] = list(sorted(values))
    return golden_dict


def generateGoldenEntities(entitiesJson):
    '''
    [(2, 3, "entity_type")]
    '''
    golden_list = []
    for entityJson in entitiesJson:
        start = entityJson["start"]
        if start >= CUTOFF:
            continue
        end = min(entityJson["end"], CUTOFF)
        etype = entityJson["entity-type"].split(":")[0]
        golden_list.append((start, end, etype))
    return golden_list


class DependencyParser(object):
    def __init__(self, sent):
        self.sent = sent
        self.stanford = StanfordCoreNLP('http://localhost:9001')
        self.properties = {'annotators': 'tokenize,ssplit,pos,lemma,ner,depparse,parse', 'outputFormat': 'json'}
        #call(["java -mx4g -cp '*' edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9001 -timeout 15000"])

    def find_dep(self):
        words_dict = {}
        offset_start_dic = {}
        offset_end_dic = {}
        indices = []
        with open("test.txt.out") as file:
            lines = file.readlines()
            flag_tok = False
            flag_dep = False
            tokens = []
            char_offset_beg = []
            char_offset_end = []
            pos_list = []
            dep_triples = []
            indices = []
            i = 1
            for line in lines:
                line = line.rstrip()
                if line == "":
                    flag_tok = False
                    flag_dep = False

                if flag_tok:
                    if len(line.split(" "))==4:
                        word = line.split(" ")[0].split("=")[1]
                        char_beg = line.split(" ")[1].split("=")[1]
                        char_end = line.split(" ")[2].split("=")[1]
                        pos = line.split(" ")[3].split("=")[1][:-1]
                    elif len(line.split(" "))==7:
                        word = line.split(" ")[0].split("=")[1]
                        char_beg = line.split(" ")[4].split("=")[1]
                        char_end = line.split(" ")[5].split("=")[1]
                        pos = line.split(" ")[6].split("=")[1][:-1]
                    else:
                        word = line.split(" ")[0].split("=")[1]
                        id_ = line.split(" ")[1].split("=")[1]
                        char_beg = line.split(" ")[2].split("=")[1]
                        char_end = line.split(" ")[3].split("=")[1]
                        pos = line.split(" ")[4].split("=")[1][:-1]

                    tokens.append(word)
                    char_offset_beg.append(int(char_beg))
                    char_offset_end.append(int(char_end))
                    pos_list.append(pos)
                    indices.append(i)
                    i += 1

                if flag_dep:
                    dep_rel = line.split("(", 1)[0]
                    parts = line.split("(", 1)[1].split(", ")

                    gov = int(parts[0].split("-")[-1])-1
                    dep = int(parts[1].split("-")[-1][:-1]) - 1

                    triple = dep_rel + "/dep=" + str(dep) + "/gov=" + str(gov)
                    dep_triples.append(triple)

                if line == "Tokens:":
                    flag_tok = True

                if line == "Dependency Parse (enhanced plus plus dependencies):":
                    flag_dep = True

        for i, word in enumerate(tokens):
            offset_start_dic.update({char_offset_beg[i]: indices[i]-1})
            offset_end_dic.update({char_offset_end[i]-1: indices[i]})
            words_dict.update({indices[i]-1: word})

        return dep_triples, tokens, pos_list, char_offset_beg, char_offset_end, offset_start_dic, offset_end_dic, words_dict

    def find_real_dep(self, sent, lang):
        os.system("echo '"+sent+"' > test.txt")
        if lang == "Arabic":
            os.system("java -mx3g -cp '*' edu.stanford.nlp.pipeline.StanfordCoreNLP -props "
                      "StanfordCoreNLP-arabic.properties -annotators tokenize,ssplit,pos,depparse -file test.txt")
        elif lang == "Chinese":
            os.system("java -mx3g -cp '*' edu.stanford.nlp.pipeline.StanfordCoreNLP -props "
                      "StanfordCoreNLP-chinese.properties -annotators tokenize,ssplit,pos,depparse -file test.txt")
        elif lang == "Spanish":
            os.system("java -mx3g -cp '*' edu.stanford.nlp.pipeline.StanfordCoreNLP -props "
                      "StanfordCoreNLP-spanish.properties -annotators tokenize,ssplit,pos,depparse -file wikipedia_ar.txt")
        else:
            os.system("java -mx3g -cp '*' edu.stanford.nlp.pipeline.StanfordCoreNLP -props "
                      "StanfordCoreNLP-english.properties -annotators tokenize,ssplit,pos,depparse -file wikipedia_ar.txt")

        dep_triples, tokens, pos_list, char_offset_beg, char_offset_end, offset_start_dic, offset_end_dic, words_dict = self.find_dep()

        return [], dep_triples, tokens, tokens, pos_list, offset_start_dic, offset_end_dic, words_dict

    def find_dep_words_pos_offsets(self, sent):
        output = self.stanford.annotate(sent, properties=self.properties)
        penn_treebank = output['sentences'][0]['parse'].replace("\n", "")
        triples = []
        for part in output['sentences'][0]['enhancedPlusPlusDependencies']:
            triples.append(part['dep']+"/dep="+str(part['dependent']-1)+"/gov="+str(part['governor']-1))

        words = []
        lemmas = []
        pos_tags = []
        words_dict = {}
        offset_start_dic = {}
        offset_end_dic = {}
        for i, word in enumerate(output['sentences'][0]['tokens']):
            words.append(word["word"])
            lemmas.append(word["lemma"])
            pos_tags.append(word["pos"])
            offset_start_dic.update({word["characterOffsetBegin"]: word["index"]-1})
            offset_end_dic.update({word["characterOffsetEnd"]-1: word["index"]})
            words_dict.update({word["index"]-1: word["word"]})

        return penn_treebank, triples, words, lemmas, pos_tags, offset_start_dic, offset_end_dic, words_dict


class Embeddings(object):
    def __init__(self, emb_filename, emb_type, vocab, dim_word):
        self.emb_filename = emb_filename
        self.dim = dim_word
        self.trimmed_filename = ".".join(emb_filename.split(".")[:-1]) + "_trimmed.npz"
        if emb_type == "fasttext" or emb_type == "glove":
            self.embed_vocab = self.get_emb_vocab()
        else:
            self.embed_vocab = self.get_multi_emb_vocab()
        if not os.path.isfile(self.trimmed_filename):
            if emb_type == "fasttext":
                self.export_trimmed_fasttext_vectors(vocab)
                self.embed_vocab = self.get_emb_vocab()
            elif emb_type == "glove":
                self.export_trimmed_glove_vectors(vocab)
                self.embed_vocab = self.get_emb_vocab()
            else:
                self.export_trimmed_multi_vectors(vocab)
                self.embed_vocab = self.get_multi_emb_vocab()

    def export_trimmed_fasttext_vectors(self, vocab):
        """Saves fasttext monolingual vectors in numpy array

        Args:
            vocab: dictionary vocab[word] = index

        """
        embeddings = np.zeros([len(vocab), self.dim])
        with open(self.emb_filename) as f:
            next(f)
            for line in f:
                line = line.strip().split(' ')
                word = line[0]
                embedding = [float(x) for x in line[1:]]
                if word in vocab:
                    word_idx = vocab[word]
                    embeddings[word_idx] = np.asarray(embedding)

        np.savez_compressed(self.trimmed_filename, embeddings=embeddings)

    def export_trimmed_glove_vectors(self, vocab):
        """Saves glove vectors in numpy array

        Args:
            vocab: dictionary vocab[word] = index

        """
        embeddings = np.zeros([len(vocab), self.dim])
        with open(self.emb_filename) as f:
            for line in f:
                line = line.strip().split(' ')
                word = line[0]
                embedding = [float(x) for x in line[1:]]
                if word in vocab:
                    word_idx = vocab[word]
                    embeddings[word_idx] = np.asarray(embedding)

        np.savez_compressed(self.trimmed_filename, embeddings=embeddings)

    def export_trimmed_multi_vectors(self, vocab):
        """Saves glove vectors in numpy array

        Args:
            vocab: dictionary vocab[word] = index

        """
        embeddings = np.zeros([len(vocab), self.dim])
        with open(self.emb_filename) as f:
            for line in f:
                line = line.strip().split(' ')
                word = line[0].split("_")[1]
                embedding = [float(x) for x in line[1:]]
                if word in vocab:
                    word_idx = vocab[word]
                    embeddings[word_idx] = np.asarray(embedding)

        np.savez_compressed(self.trimmed_filename, embeddings=embeddings)

    def get_trimmed_vectors(self):
        """
        Args:
            filename: path to the npz file

        Returns:
            matrix of embeddings (np array)
        """
        try:
            with np.load(self.trimmed_filename) as data:
                print(data["embeddings"])
                return data["embeddings"]

        except IOError:
            raise Exception("Could not find or load file!!", self.trimmed_filename)

    def get_emb_vocab(self):
        """Load vocab from file

        Returns:
            vocab: set() of strings
        """
        print("Building vocab...")
        vocab = set()
        with open(self.emb_filename) as f:
            lines = f.readlines()
            for line in lines[1:]:
                word = line.strip().split(' ')[0]
                vocab.add(word)
        print("- done. {} tokens".format(len(vocab)))

        return vocab

    def get_multi_emb_vocab(self):
        """Load vocab from file

        Returns:
            vocab: set() of strings
        """
        print("Building vocab...")
        vocab = set()
        with open(self.emb_filename) as f:
            lines = f.readlines()
            for line in lines:
                word = line.strip().split(' ')[0].split("_")[1]
                vocab.add(word)
        print("- done. {} tokens".format(len(vocab)))

        return vocab


class CoNLLDataset(object):
    """Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags

    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```

    """
    def __init__(self, filename, lang, mode, processing_word=None, processing_tag=None,
                 max_iter=None):
        """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield

        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None
        self.lang = lang
        self.mode = mode

    def __iter__(self):
        niter = 0
        with open(self.filename) as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if len(line) == 0 or line.startswith("-DOCSTART-"):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags
                        words, tags = [], []
                else:
                    ls = line.split("\t")
                    word, tag = ls[0], ls[-1]
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    words += [word]
                    tags += [tag]

    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


class CoNLLDatasetJoint(object):
    """Joint Version of CoNLLDataset Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        trig_tags: list of raw trigger tags
        arg_tags: list of raw argument tags

    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, trig_tags, arg_tags in data:
            pass
        ```

    """
    def __init__(self, filename, processing_word=None, processing_pos_tag=None, processing_trig=None,
                 processing_event_tag=None, max_iter=100):
        """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield

        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_pos_tag = processing_pos_tag
        self.processing_trig = processing_trig
        self.processing_event_tag = processing_event_tag
        self.max_iter = max_iter
        self.length = None

    def __iter__(self):
        niter = 0
        with open(self.filename) as f:
            data = json.load(f)
            for sent in data:
                words, pos_tags, trig_tags, arg_tags, ent_args = [], [], [], [], []
                niter += 1
                if self.max_iter is not None and niter > self.max_iter:
                    break

                ### Processing of Words
                if self.processing_word is not None:
                    for word in sent["words"]:
                        words += [self.processing_word(word)]
                else:
                    words += sent["words"]

                ### Processing of POS-TAGS
                if self.processing_pos_tag is not None:
                    for pos_tag in sent["pos-tags"]:
                        pos_tags += [self.processing_pos_tag(pos_tag)]
                else:
                    pos_tags += sent["pos-tags"]

                ### Processing of Triggers
                triggers = generateTriggerLabelList(sent["golden-event-mentions"], len(sent["words"]))
                if self.processing_trig is not None:
                    for trig in triggers:
                        trig_tags += [self.processing_trig(trig)]
                else:
                    trig_tags += triggers

                ### Processing of Event Arguments
                events = generateGoldenEvents(sent["golden-event-mentions"])
                if self.processing_event_tag is not None:
                    arg_tags += [self.processing_event_tag(events)]
                else:
                    arg_tags += [events]

                ### Entities
                ent_args += generateGoldenEntities(sent["golden-entity-mentions"])

                yield words, pos_tags, trig_tags, arg_tags, ent_args

    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def get_vocabs(datasets, joint=None):
    """Build vocabulary from an iterable of datasets objects

    Args:
        datasets: a list of dataset objects

    Returns:
        a set of all the words in the dataset

    """
    print("Building vocab...")
    if joint is not None:
        vocab_words = set()
        vocab_pos_tags = set()
        vocab_trig_tags = set()
        vocab_arg_tags = set()
        vocab_ent_tags = set()

        vocab_words.add(UNK)
        vocab_pos_tags.add(UNK)
        vocab_trig_tags.add(UNK)
        vocab_arg_tags.add(UNK)
        vocab_ent_tags.add(UNK)

        for dataset in datasets:
            for words, pos_tags, trig_tags, arg_tags, ent_args in dataset:
                vocab_words.update(words)
                vocab_pos_tags.update(pos_tags)
                vocab_trig_tags.update(trig_tags)
                for i in range(0, len(arg_tags)):
                    for event in arg_tags[i]:
                        for (st, ed, role) in arg_tags[i][event]:
                            vocab_arg_tags.add(role)

                for ent in ent_args:
                    vocab_ent_tags.add(ent[2])

        print("- done. {} tokens".format(len(vocab_words)))
        return vocab_words, vocab_pos_tags, vocab_trig_tags, vocab_arg_tags, vocab_ent_tags
    else:
        vocab_words = set()
        vocab_tags = set()
        for dataset in datasets:
            for words, tags in dataset:
                vocab_words.update(words)
                vocab_tags.update(tags)
        print("- done. {} tokens".format(len(vocab_words)))
        return vocab_words, vocab_tags


def get_char_vocab(datasets, joint=None):
    """Build char vocabulary from an iterable of datasets objects

    Args:
        dataset: a iterator yielding tuples (sentence, tags)

    Returns:
        a set of all the characters in the dataset

    """
    vocab_char = set()
    for dataset in datasets:
        if joint is not None:
            for words, pos_tags, trig_tags, arg_tags, ent_args in dataset:
                for word in words:
                    vocab_char.update(word)
        else:
            for words, _ in dataset:
                for word in words:
                    vocab_char.update(word)

    return vocab_char


def get_processing_word(vocab_words=None, vocab_chars=None, lowercase=False, chars=False, allow_unk=True):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                    #print("len(vocab_words):", len(vocab_words))
                else:
                    raise Exception("Unknow key is not allowed. Check that " \
                                    "your vocab (tags?) is correct =>"+ str(len(vocab_words)))

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars is True:
            return char_ids, word
        else:
            return word

    return f


def get_processing_arg_role(vocab_words=None, allow_unk=True):
    def f(arg_tags):
        if vocab_words is not None:
            args_tags_ids = {}
            for event in arg_tags:
                for (st, ed, role) in arg_tags[event]:
                    if role in vocab_words:
                        role_id = vocab_words[role]
                    else:
                        if allow_unk:
                            role_id = vocab_words[UNK]
                        else:
                            raise Exception("Unknown key is not allowed. Check that "\
                                            "your vocab (tags?) is correct =>" + str(len(vocab_words)))

                    if event not in args_tags_ids:
                        args_tags_ids.update({event: [(st, ed, role_id)]})
                    else:
                        args = args_tags_ids[event]
                        args.append((st, ed, role_id))
                        args_tags_ids.update({event: args})

            return args_tags_ids
        else:
            return arg_tags
    return f


def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    try:
        d = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx

    except IOError:
        raise Exception("Could not find or load file!!", filename)
    return d


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, input_mask, segment_ids, sequence_length = _pad_sequences(sequences,
                                                          pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, im, si, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, input_mask, segment_ids,  _ = _pad_sequences(sequence_padded,
                                            [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, input_mask, segment_ids, _ = _pad_sequences(sequence_length, 0,
                                            max_length_sentence)

    return sequence_padded, input_mask, segment_ids,  sequence_length


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, input_mask, segment_ids, sequence_length = [], [], [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        inp_mask = [1] * len(seq[:max_length]) + [0] * max(max_length - len(seq), 0)
        seg_id = [0] * len(seq[:max_length]) + [0] * max(max_length - len(seq), 0)

        sequence_padded += [seq_]
        input_mask += [inp_mask]
        segment_ids += [seg_id]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, input_mask, segment_ids, sequence_length


def minibatches(data, minibatch_size, mode, joint=None):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    if joint is not None:
        x_batch, pos_batch, trig_batch, arg_batch, ent_batch = [], [], [], [], []
        for lang in data:
            for (words, pos_tags, trig_tags, arg_tags, ent_args) in data[lang]:

                if len(x_batch) == minibatch_size:
                    yield x_batch, pos_batch, trig_batch, arg_batch, ent_batch
                    x_batch, pos_batch, trig_batch, arg_batch, ent_batch = [], [], [], [], []

                if type(words[0]) == tuple:
                    words = zip(*words)
                x_batch += [words]
                pos_batch += [pos_tags]
                trig_batch += [trig_tags]
                arg_batch += [arg_tags]
                ent_batch += [ent_args]

        if len(x_batch) != 0:
            yield x_batch, pos_batch, trig_batch, arg_batch, ent_batch

    else:
        if mode =="train":
            x_batch, y_batch = [], []
            for lang in data:
                for (x, y) in data[lang]:
                    if len(x_batch) == minibatch_size:
                        yield x_batch, y_batch
                        x_batch, y_batch = [], []

                    if type(x[0]) == tuple:
                        x = zip(*x)
                    x_batch += [x]
                    y_batch += [y]

            if len(x_batch) != 0:
                yield x_batch, y_batch
        else:
            x_batch, y_batch = [], []
            for (x, y) in data:
                if len(x_batch) == minibatch_size:
                    yield x_batch, y_batch
                    x_batch, y_batch = [], []

                if type(x[0]) == tuple:
                    x = zip(*x)
                x_batch += [x]
                y_batch += [y]

            if len(x_batch) != 0:
                yield x_batch, y_batch


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


if __name__ == '__main__':
    #sent = "The skeleton of a second baby has been found at a rural Wisconsin home where a 22-year-old woman's dead" \
    #       " infant was discovered in a blue plastic container June 8, officials said Monday."

    sent = "واشنطن حيث رحبت الإدارة الأمريكية بهذه الخطوة" #"以军方没有提到以军伤亡情况""
    #sent = "Paul, as I understand your definition of a political -- of a professional politician based on that is somebody who is elected to public office."

    dp = DependencyParser(sent)
    penn_treebank, triples, words, lemmas, pos_tags, offset_start_dic, offset_end_dic, words_dict \
        = dp.find_dep_words_pos_offsets(sent)

    print("penn_treebank:", penn_treebank)

    for trip in triples:
        print(trip)

    print("words:", words)
    for pos in pos_tags:
        print(pos)
