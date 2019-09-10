import xml.etree.ElementTree as ET
from data_utils import *
from ace_ere_utils import *
from tqdm import tqdm
from get_args import *
import json
from nltk import sent_tokenize  # Setup stanford corenlp tools with NLTK

import os


def extract_doc_info(file_name):
    lines = []
    with open(file_name+".sgm", "r") as file:
        for line in file:
            lines.append(line.replace("&", "&amp;"))

    with open(file_name+"_modified.sgm", "w") as file:
        for line in lines:
            file.write(line)

    tree = ET.parse(file_name+"_modified.sgm", ET.XMLParser(encoding='utf-8'))
    root = tree.getroot()
    for docid in root.iter('DOCID'):
        doc_id = docid.text

    try:
        for doctype in root.iter('DOCTYPE'):
            source = doctype.attrib['SOURCE']
    except:
        source = None

    for datetime in root.iter('DATETIME'):
        datetime = datetime.text

    for text in root.iter('TEXT'):
        turns = text.text

    return ACEDocument(doc_id, source, datetime, turns)


def add_non_events(sents_events, raw_path, splits_file_path, mode, language):
    non_events_sent = []
    non_events_json = []
    print("splits_file_path:", splits_file_path)
    with open(splits_file_path + mode) as file:
        files = file.read().splitlines()
    for file in tqdm(files):
        doc = extract_doc_info(raw_path + language + "/" + file)

        offset, length = 0, 0
        for sent in sent_tokenize(doc.text):
            flag = False
            while doc.text[offset] != sent[0]:
                offset += 1

            length = len(sent)
            offset += length

            for sent2 in sents_events:
                for event in sent2["golden-event-mentions"]:
                    start = event["trigger"]["start"]
                    end = event["trigger"]["end"]
                    if offset < start and offset+length > end:
                        flag = True
            if not flag:
                non_events_sent.append(sent)

    for sent in non_events_sent:
        depParser = DependencyParser(sent)
        penn_treebank, triples, words, pos_tags, offset_start_dic, offset_end_dic, words_dict = \
            depParser.find_dep_words_pos_offsets(sent)
        non_events_json.append({"sentence": sent, "penn_treebank": penn_treebank, "stanford-colcc": triples,
                                "words": words, "pos-tags": pos_tags, "golden-entity-mentions": [],
                                "golden-event-mentions": []})

    return non_events_sent


if __name__ == '__main__':

    args = get_arguments()

    if not os.path.exists(args.pre_dir):
        os.makedirs(args.pre_dir)

    data_path_dir = args.json_dir
    raw_path = args.root_dir + args.data_ace_path

    events_lang = {}
    for language in args.languages.split(","):
        splits_file_path = args.jmee_splits + language + "/"

        files_num = 0
        for split in ["train", "dev", "test"]:
            with open(data_path_dir+split+".json") as file:
                data_json = json.load(file)
            sents_events = []
            for sent in data_json:
                sents_events.append(sent)

            non_events_json = add_non_events(sents_events, raw_path, splits_file_path, split, language)

            for sent in non_events_json:
                data_json.append(sent)

            with open(data_path_dir+split+"_neg_eg.json", 'w') as outfile:
                json.dump(data_json, outfile, indent=4, sort_keys=True)


