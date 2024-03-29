import json
import os
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher

from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from get_args import *
from DataModule.ace_ere_utils import *
from DataModule.data_utils import *


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z


class DataPreprocessor(object):
    def __init__(self, data_path, splits_file_path, language):
        self.data_path = data_path
        self.splits_file_path = splits_file_path
        self.language = language

    def extract_entity_info(self, entity, scope_start, scope_end, sent, words, offset_start_dic, offset_end_dic):
        entity_id = entity.attrib["ID"]
        entity_type = entity.attrib["TYPE"] + ":" + entity.attrib["SUBTYPE"]
        entities = []
        for mention in entity.iter('entity_mention'):
            phrase_type = mention.attrib["LDCTYPE"]
            for child in mention:
                if child.tag == "extent":
                    for chil2 in child:
                        text = chil2.text
                        start = int(chil2.attrib["START"])
                        end = int(chil2.attrib["END"])

            if scope_start <= start and scope_end >= end:
                try:
                    try:
                        start_off = max(offset_start_dic[k] for k in offset_start_dic if k <= start - scope_start)
                    except:
                        start_off = offset_start_dic[list(offset_start_dic.keys())[0]]
                    try:
                        end_off = max(offset_end_dic[k] for k in offset_end_dic if k <= end - scope_start)
                    except:
                        end_off = offset_end_dic[list(offset_end_dic.keys())[0]] + 1

                    if end_off == start_off:
                        end_off += 1
                    ent = Entity(entity_id, text.replace("\n", " "), entity_type, phrase_type, start_off, end_off)
                    entities.append(ent)
                except:
                    continue

        return entities

    def extract_timex2_info(self, entity, scope_start, scope_end, sent, words, offset_start_dic, offset_end_dic):
        entity_id = entity.attrib["ID"]
        phrase_type = "TIM"
        entity_type = "TIM:time"
        entities = []
        for mention in entity.iter('timex2_mention'):
            for child in mention:
                if child.tag == "extent":
                    for chil2 in child:
                        text = chil2.text
                        start = int(chil2.attrib["START"])
                        end = int(chil2.attrib["END"])

            if scope_start <= start and scope_end >= end:
                try:
                    try:
                        start_off = max(offset_start_dic[k] for k in offset_start_dic if k <= start - scope_start)
                    except:
                        start_off = offset_start_dic[list(offset_start_dic.keys())[0]]
                    try:
                        end_off = max(offset_end_dic[k] for k in offset_end_dic if k <= end - scope_start)
                    except:
                        end_off = offset_end_dic[list(offset_end_dic.keys())[0]] + 1

                    if end_off == start_off:
                        end_off += 1
                    ent = Entity(entity_id, text.replace("\n", " "), entity_type, phrase_type, start_off, end_off)
                    entities.append(ent)
                except:
                    continue

        return entities

    def extract_value_info(self, entity, scope_start, scope_end, sent, words, offset_start_dic, offset_end_dic):
        entity_id = entity.attrib["ID"]
        phrase_type = "VALUE"
        if "SUBTYPE" in entity.attrib:
            entity_type = entity.attrib["TYPE"] + ":" + entity.attrib["SUBTYPE"]
        else:
            entity_type = entity.attrib["TYPE"] + ":other"
        entities = []
        for mention in entity.iter('value_mention'):
            for child in mention:
                if child.tag == "extent":
                    for chil2 in child:
                        text = chil2.text
                        start = int(chil2.attrib["START"])
                        end = int(chil2.attrib["END"])

            if scope_start <= start and scope_end >= end:
                try:
                    try:
                        start_off = max(offset_start_dic[k] for k in offset_start_dic if k <= start - scope_start)
                    except:
                        start_off = offset_start_dic[list(offset_start_dic.keys())[0]]
                    try:
                        end_off = max(offset_end_dic[k] for k in offset_end_dic if k <= end - scope_start)
                    except:
                        end_off = offset_end_dic[list(offset_end_dic.keys())[0]] + 1

                    if end_off == start_off:
                        end_off += 1
                    ent = Entity(entity_id, text.replace("\n", " "), entity_type, phrase_type, start_off, end_off)
                    entities.append(ent)
                except:
                    continue

        return entities

    def extract_event_info(self, root, event, lang):
        event_id = event.attrib["ID"]
        event_type = event.attrib["TYPE"]
        subtype = event.attrib["SUBTYPE"]
        modality = event.attrib["MODALITY"]
        polarity = event.attrib["POLARITY"]
        genericity = event.attrib["GENERICITY"]
        tense = event.attrib["TENSE"]

        ## Looking at event mentions
        i = 0
        for mention in event.iter('event_mention'):
            if i == 0:
                mention_id = mention.attrib["ID"]
                for child in mention:
                    if child.tag == "extent":
                        for chil2 in child:
                            extent = chil2.text.replace("&", "&amp;")
                            extent_start = int(chil2.attrib["START"])
                            extent_end = int(chil2.attrib["END"])

                    ## SCOPE USED AS SENTENCE
                    elif child.tag == "ldc_scope":
                        for chil2 in child:
                            scope = chil2.text
                            scope_start = int(chil2.attrib["START"])
                            scope_end = int(chil2.attrib["END"])
                            sent = Sentence(scope, scope_start, scope_end)
                            depParser = DependencyParser(sent)
                            penn_treebank, triples, words, lemmas, pos_tags, offset_start_dic, offset_end_dic, \
                            words_dict = depParser.find_real_dep(scope, lang)

                    ## TRIGGER EXTRACTION
                    elif child.tag == "anchor":
                        for chil2 in child:
                            trig_text = chil2.text
                            start = int(chil2.attrib["START"]) - scope_start
                            end = int(chil2.attrib["END"]) - scope_start

                            try:

                                try:
                                    trig_start = max(offset_start_dic[k] for k in offset_start_dic if k <= start) #offset_start_dic[start]
                                except:
                                    trig_start = offset_start_dic[list(offset_start_dic.keys())[0]]
                                try:
                                    trig_end = max(offset_end_dic[k] for k in offset_end_dic if k <= end)
                                except:
                                    trig_end = offset_end_dic[list(offset_start_dic.keys())[0]] + 1

                            except:
                                print("Problematic sentence:", sent)
                                print("words: ", words)
                                print("offset_start_dic:", offset_start_dic)
                                print("offset_end_dic:", offset_end_dic)
                                print("trig_text:", trig_text)
                                print("start:", start)
                                print("end:", end)
                                trig_start = -1
                                trig_end = -1
                                continue

                            if trig_start == trig_end:
                                trig_end += 1

                ## Looking at entity mentions with that same event
                entities = []
                ent_id_role_dict = {}
                for entity in root.iter('entity'):
                    ents = self.extract_entity_info(entity, scope_start, scope_end, sent, words, offset_start_dic,
                                                    offset_end_dic)

                    entities.extend(ents)
                    if len(ents) > 0:
                        ent_id_role_dict.update({ents[0].id_: ents[0].entity_type})

                for entity in root.iter('timex2'):
                    ents = self.extract_timex2_info(entity, scope_start, scope_end, sent, words, offset_start_dic,
                                                    offset_end_dic)

                    entities.extend(ents)
                    if len(ents) > 0:
                        ent_id_role_dict.update({ents[0].id_: ents[0].entity_type})

                for entity in root.iter('value'):
                    ents = self.extract_value_info(entity, scope_start, scope_end, sent, words, offset_start_dic,
                                                    offset_end_dic)

                    entities.extend(ents)
                    if len(ents) > 0:
                        ent_id_role_dict.update({ents[0].id_: ents[0].entity_type})

                arguments = []
                for argument in mention.iter('event_mention_argument'):
                    arg_id = argument.attrib["REFID"]
                    role = argument.attrib["ROLE"]
                    for child in argument:
                        for chil2 in child:
                            arg_text = chil2.text
                            start = int(chil2.attrib["START"]) - scope_start
                            end = int(chil2.attrib["END"]) - scope_start

                            try:
                                try:
                                    arg_start = max(offset_start_dic[k] for k in offset_start_dic if k <= start)
                                except:
                                    arg_start = offset_start_dic[list(offset_start_dic.keys())[0]]
                                try:
                                    arg_end = max(offset_end_dic[k] for k in offset_end_dic if k <= end)
                                except:
                                    arg_end = offset_end_dic[list(offset_start_dic.keys())[0]] + 1

                                if arg_start == arg_end:
                                    arg_end += 1

                                if "-".join(arg_id.split("-")[:-1]) in ent_id_role_dict:
                                    type_ = ent_id_role_dict["-".join(arg_id.split("-")[:-1])]
                                elif arg_id in ent_id_role_dict:
                                    type_ = ent_id_role_dict[arg_id]
                                else:
                                    type_ = "--"
                                arg = Argument(arg_id, arg_text, role, arg_start, arg_end,type_)

                                arguments.append(arg)
                            except:
                                print("Problematic sentence:", sent)
                                print("words: ", words)
                                print("offset_start_dic:", offset_start_dic)
                                print("offset_end_dic:", offset_end_dic)
                                print("argument:", arg_text)
                                continue
            i += 1
        ev = Event(event_id, mention_id, event_type, subtype, modality, polarity, genericity, tense, extent,
                   extent_start, extent_end, scope, scope_start, scope_end, trig_text, trig_start, trig_end,
                   arguments, entities)

        return sent, ev, penn_treebank, triples, words, pos_tags

    def extract_event_info_bio(self, root, event, lang):
        event_id = event.attrib["ID"]
        event_type = event.attrib["TYPE"]
        subtype = event.attrib["SUBTYPE"]
        modality = event.attrib["MODALITY"]
        polarity = event.attrib["POLARITY"]
        genericity = event.attrib["GENERICITY"]
        tense = event.attrib["TENSE"]

        ## Looking at event mentions
        for mention in event.iter('event_mention'):
            mention_id = mention.attrib["ID"]
            for child in mention:
                if child.tag == "extent":
                    for chil2 in child:
                        extent = chil2.text.replace("&", "&amp;")
                        extent_start = int(chil2.attrib["START"])
                        extent_end = int(chil2.attrib["END"])

                ## SCOPE USED AS SENTENCE
                elif child.tag == "ldc_scope":
                    for chil2 in child:
                        scope = chil2.text
                        scope_start = int(chil2.attrib["START"])
                        scope_end = int(chil2.attrib["END"])
                        sent = Sentence(scope, scope_start, scope_end)
                        depParser = DependencyParser(sent)

                        penn_treebank, triples, words, pos_tags, offset_start_dic, offset_end_dic, words_dict = \
                            depParser.find_real_dep(scope, lang)

                ## TRIGGER EXTRACTION
                elif child.tag == "anchor":
                    for chil2 in child:
                        trig_text = chil2.text
                        trig_start = int(chil2.attrib["START"]) - scope_start
                        trig_end = int(chil2.attrib["END"]) - scope_start

            ## Looking at entity mentions with that same event
            entities = []
            ent_id_role_dict = {}
            for entity in root.iter('entity'):
                ents = self.extract_entity_info(entity, extent_start, extent_end, sent, words, offset_start_dic,
                                                offset_end_dic)
                entities.extend(ents)
                if len(ents) > 0:
                    ent_id_role_dict.update({ents[0].id_: ents[0].entity_type})

            arguments = []
            for argument in mention.iter('event_mention_argument'):
                arg_id = argument.attrib["REFID"]
                role = argument.attrib["ROLE"]
                for child in argument:
                    for chil2 in child:
                        arg_text = chil2.text
                        arg_start = int(chil2.attrib["START"]) - scope_start
                        arg_end = int(chil2.attrib["END"]) - scope_start

                        if "-".join(arg_id.split("-")[:-1]) in ent_id_role_dict:
                            type_ = ent_id_role_dict["-".join(arg_id.split("-")[:-1])]
                        elif arg_id in ent_id_role_dict:
                            type_ = ent_id_role_dict[arg_id]
                        else:
                            type_ = "--"
                        arg = Argument(arg_id, arg_text, role, arg_start, arg_end, type_)

                        arguments.append(arg)

        ev = Event(event_id, mention_id, event_type, subtype, modality, polarity, genericity, tense, extent,
                   extent_start, extent_end, scope, scope_start, scope_end, trig_text, trig_start, trig_end,
                   arguments, entities)

        return sent, ev, penn_treebank, triples, words, pos_tags

    def extract_doc_info(self, file_name):
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

    def extract_from_xml(self, mode, lang):
        events = {}
        non_events_sent = []
        with open(self.splits_file_path + mode) as file:
            files = file.read().splitlines()

        files_processed = 0
        for file in tqdm(files):
            # Get the event + argument annotation
            file_name = self.data_path + self.language + "/" + file + ".apf.xml"
            files_processed += 1
            tree = ET.parse(file_name, ET.XMLParser(encoding='utf-8'))
            root = tree.getroot()

            for event in root.iter('event'):
                sent, ev, penn_treebank, triples, words, pos_tags = self.extract_event_info(root, event, lang)
                if sent.text not in events:
                    events.update({sent.text: {"events": [ev], "penn_treebank": penn_treebank, "triples": triples,
                                               "words": words, "pos_tags": pos_tags}})
                else:
                    ev_list = events[sent.text]["events"]
                    ev_list.append(ev)
                    events[sent.text]["events"] = ev_list

            doc = self.extract_doc_info(self.data_path + self.language + "/" + file)
            sents = sent_tokenize(doc.text)
            for sent in sents:
                flag = False
                for sent2 in events:
                    if similar(sent, sent2) > 0.95:
                        flag = True
                if not flag:
                    non_events_sent.append(sent)

        return events, non_events_sent, files_processed

    def using_jmee_split(self):
        files_splits = {}
        with open(self.splits_file_path+"training") as file:
            files_splits.update({"train": file.readlines()})

        with open(self.splits_file_path+"dev") as file:
            files_splits.update({"dev": file.readlines()})

        with open(self.splits_file_path+"test") as file:
            files_splits.update({"test": file.readlines()})

        return files_splits

    def save_jmee_json(self, events, lang, mode, non_events_sent, pre_dir, include_neg):
        sent_json = []
        data_json = []
        for sent in tqdm(events):
            sent_json.append(sent)
            data_sub = {"golden-event-mentions": []}
            entities_unique = {}
            for event in events[sent]["events"]:
                event_info = {}
                event_info["trigger"] = {"start": event.trig_start, "end": event.trig_end, "text": event.trig_text}
                event_info["arguments"] = []
                for arg in event.arguments:
                    arg_info = {"start": arg.start, "role": arg.role, "end": arg.end, "text": arg.text}
                    event_info["arguments"].append(arg_info)

                event_info["id"] = event.event_id
                event_info["event_type"] = event.type_ + ":" + event.subtype
                data_sub["golden-event-mentions"].append(event_info)

                # Loading entities for that event and adding it to the list of entities
                for entity in event.entities:
                    entities_unique.update({entity.id_+"||"+entity.text: entity})

            data_sub["golden-entity-mentions"] = []
            for entity_id in entities_unique.keys():
                entity_info = {"phrase-type": entities_unique[entity_id].phrase_type,
                               "end": entities_unique[entity_id].end, "text": entities_unique[entity_id].text,
                               "entity-type": entities_unique[entity_id].entity_type,
                               "start": entities_unique[entity_id].start, "id": entities_unique[entity_id].id_}
                data_sub["golden-entity-mentions"].append(entity_info)

            data_json.append({"penn_treebank": events[sent]["penn_treebank"], "stanford-colcc": events[sent]["triples"],
                              "words": events[sent]["words"], "pos-tags": events[sent]["pos_tags"],
                              "golden-entity-mentions": data_sub["golden-entity-mentions"], "sentence": sent.replace("\n", " "),
                              "golden-event-mentions": data_sub["golden-event-mentions"]})

        if not os.path.exists(pre_dir + lang + "/"):
            os.makedirs(pre_dir + lang + "/")

        if include_neg:
            for sent in non_events_sent:
                depParser = DependencyParser(sent)

                penn_treebank, triples, words, lemmas, pos_tags, offset_start_dic, offset_end_dic, words_dict = \
                    depParser.find_real_dep(sent, lang)
                data_json.append({ "sentence": sent, "penn_treebank": penn_treebank, "stanford-colcc": triples,
                                   "words": words, "pos-tags": pos_tags, "golden-entity-mentions": [],
                                   "golden-event-mentions": []})

            path = mode+"_with_neg_eg.json"
        else:
            path = mode+"_wout_neg_eg.json"

        file_path = pre_dir + lang + "/" + path
        print("Saving json file in ", file_path)
        with open(file_path, 'w') as outfile:
            json.dump(data_json, outfile, indent=4, sort_keys=True)

        return data_json

    def get_triggers_bio(self, events, non_events_sent, lang, mode, pre_dir, include_neg):
        new_trig_dict = {}
        for sent in events.keys():
            triggers = []
            new_triggers = []
            for event in events[sent]["events"]:
                triggers.append(Trigger(event.trig_start, event.trig_text, event.trig_end, event.event_id,
                                        event.type_+":"+event.subtype))
            triggers.sort(key=lambda x: x.start, reverse=False)
            end = 0
            for trig in triggers:
                if trig.start > end:
                    new_triggers.append(trig)
                end = trig.end
            new_trig_dict.update({sent: new_triggers})

        words_split = []
        for sent in new_trig_dict:
            start = 0
            words = []
            for event in new_trig_dict[sent]:
                end = event.start

                ### Tokenize that part that doesn't have to do with triggers and annotate each word as 'O'
                words.extend([(word, "O") for word in word_tokenize(sent[start:end])])

                ### Tokenize trigger part and annotate each word as 'B' or 'I'
                start = event.start
                end = event.end+1
                trigger_tok = word_tokenize(sent[start:end])
                flag = True
                for word in trigger_tok:
                    if flag:
                        flag = False
                        words.append((word, "B-" + event.event_type))
                    else:
                        words.append((word, "I-" + event.event_type))

                start = event.end + 1

            words.extend([(word, "O") for word in word_tokenize(sent[start:])])
            words_split.append(words)

        if include_neg:
            for sent in non_events_sent:
                words_split.append([(word, "O") for word in word_tokenize(sent)])

            path = mode+"_with_neg_eg.txt"
        else:
            path = mode+"_wout_neg_eg.txt"

        root_path = pre_dir + "TriggerIdentification/" + lang + "/"
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        with open(root_path+path, "w") as file:
            for sent in words_split:
                for word, ann in sent:
                    file.write(word + " " + ann+"\n")
                file.write("\n")

        return words_split

    def get_arguments_bio(self, events, non_events_sent, lang, mode, pre_dir, include_neg):
        new_arguments_dict = {}
        for sent in events.keys():
            new_arguments = []
            events_arg = []
            for event in events[sent]["events"]:
                events_arg.extend(event.arguments)

            events_arg.sort(key=lambda x: x.start, reverse=False)

            end = 0
            for argument in events_arg:
                if argument.start > end:
                    new_arguments.append(argument)
                end = argument.end

            new_arguments_dict.update({sent: new_arguments})

        words_split = []
        for sent in tqdm(new_arguments_dict):
            start = 0
            words = []
            for argument in new_arguments_dict[sent]:
                end = argument.start

                ### Tokenize that part that doesn't have to do with triggers and annotate each word as 'O'
                words.extend([(word, "O") for word in  word_tokenize(sent[start:end])])

                ### Tokenize trigger part and annotate each word as 'B' or 'I'
                start = argument.start
                end = argument.end+1

                arg_tok = word_tokenize(sent[start:end])
                flag = True
                for word in arg_tok:
                    if flag:
                        flag = False
                        words.append((word, "B-" + argument.role))
                    else:
                        words.append((word, "I-" + argument.role))

                start = argument.end + 1

            words.extend([(word, "O") for word in word_tokenize(sent[start:])])

            words_split.append(words)

        if include_neg:
            for sent in non_events_sent:
                words_split.append([(word, "O") for word in word_tokenize(sent)])

            path = mode+"_with_neg_eg.txt"
        else:
            path = mode+"_wout_neg_eg.txt"

        root_path = pre_dir + "ArgumentIdentification/" + lang + "/"
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        with open(root_path +path, "w") as file:
            for sent in words_split:
                for word, ann in sent:
                    file.write(word + " " + ann+"\n")
                file.write("\n")
            
        return words_split

    def get_joint_dataset(self, events, non_events_sent, lang, mode, pre_dir, include_neg):
        new_trig_arg_dict = {}
        for sent in events.keys():
            events_trig_arg = []
            for event in events[sent]["events"]:
                events_trig_arg.append(Trigger(event.trig_start, event.trig_text, event.trig_end, event.event_id,
                                        event.type_+":"+event.subtype))
                events_trig_arg.extend(event.arguments)

            events_trig_arg.sort(key=lambda x: x.start, reverse=False)

            new_trig_arg = []
            end = 0
            for trig in events_trig_arg:
                if trig.start > end:
                    new_trig_arg.append(trig)
                end = trig.end
            new_trig_arg_dict.update({sent: new_trig_arg})

        words_split = []
        trig_split = []
        arg_split = []
        for sent in tqdm(new_trig_arg_dict):
            start = 0
            word_list = []
            trig_list = []
            arg_list = []
            for ent in new_trig_arg_dict[sent]:
                end = ent.start

                ### Tokenize that part that doesn't have to do with triggers and annotate each word as 'O'
                words = word_tokenize(sent[start:end])
                word_list.extend(words)
                trig_list.extend(["O" for _ in words])
                arg_list.extend(["O" for _ in words])

                ### Tokenize trigger part and annotate each word as 'B' or 'I'
                start = ent.start
                end = ent.end+1
                tok = word_tokenize(sent[start:end])

                flag = True
                for word in tok:
                    word_list.append(word)
                    if isinstance(ent, Trigger):
                        if flag:
                            flag = False
                            trig_list.append("B-" + ent.event_type)
                        else:
                            trig_list.append("I-" + ent.event_type)
                        arg_list.append("O")

                    else:
                        if flag:
                            flag = False
                            arg_list.append("B-" + ent.role)
                        else:
                            arg_list.append("I-" + ent.role)
                        trig_list.append("O")

                start = ent.end + 1

            ## Tokenize what is left
            words = word_tokenize(sent[start:])
            word_list.extend(words)
            trig_list.extend(["O" for _ in words])
            arg_list.extend(["O" for _ in words])

            words_split.append(word_list)
            trig_split.append(trig_list)
            arg_split.append(arg_list)

        if include_neg:
            for sent in non_events_sent:
                words_split.append([(word, "O") for word in word_tokenize(sent)])

            path = mode+"_with_neg_eg.txt"
        else:
            path = mode+"_wout_neg_eg.txt"

        root_path = pre_dir + "Joint/" + lang + "/"
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        with open(root_path+path, "w") as file:
            for i, sent in enumerate(words_split):
                for j, word in enumerate(sent):
                    file.write(word + " " + trig_split[i][j] + " " + arg_split[i][j]+"\n")
                file.write("\n")

        return words_split, trig_list, arg_split

    def get_triggers_bio(self, events, non_events_sent, lang, mode, pre_dir, include_neg):
        new_trig_dict = {}
        for sent in events.keys():
            triggers = []
            new_triggers = []
            for event in events[sent]["events"]:
                triggers.append(Trigger(event.trig_start, event.trig_text, event.trig_end, event.event_id,
                                        event.type_+":"+event.subtype))
            triggers.sort(key=lambda x: x.start, reverse=False)
            end = 0
            for trig in triggers:
                if trig.start > end:
                    new_triggers.append(trig)
                end = trig.end
            new_trig_dict.update({sent: new_triggers})

        words_split = []
        for sent in new_trig_dict:
            start = 0
            words = []
            for event in new_trig_dict[sent]:
                end = event.start

                ### Tokenize that part that doesn't have to do with triggers and annotate each word as 'O'
                words.extend([(word, "O") for word in word_tokenize(sent[start:end])])

                ### Tokenize trigger part and annotate each word as 'B' or 'I'
                start = event.start
                end = event.end+1
                trigger_tok = word_tokenize(sent[start:end])
                flag = True
                for word in trigger_tok:
                    if flag:
                        flag = False
                        words.append((word, "B-" + event.event_type))
                    else:
                        words.append((word, "I-" + event.event_type))

                start = event.end + 1

            words.extend([(word, "O") for word in word_tokenize(sent[start:])])
            words_split.append(words)

        if include_neg:
            for sent in non_events_sent:
                words_split.append([(word, "O") for word in word_tokenize(sent)])

            path = mode+"_with_neg_eg.txt"
        else:
            path = mode+"_wout_neg_eg.txt"

        root_path = pre_dir + "TriggerIdentification/" + lang + "/"
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        with open(root_path+path, "w") as file:
            for sent in words_split:
                for word, ann in sent:
                    file.write(word + " " + ann+"\n")
                file.write("\n")

        return words_split


if __name__ == '__main__':

    args = get_arguments()

    languages = args.languages.split(",")

    if not os.path.exists(args.pre_dir):
        os.makedirs(args.pre_dir)

    data_path_dir = args.root_dir + args.data_ace_path

    print("args.method:", args.method)
    if args.method == "jmee":
        pre_dir = args.root_dir + args.pre_dir + "JMEE_test_new_dep/"
    else:
        pre_dir = args.root_dir + args.pre_dir + "tagging-entities/"  # "tagging-new/"

    events_lang = {}
    for language in languages:
        if language == "English" and args.split_option == "jmee":
            splits_file_path = args.jmee_splits
        else:
            splits_file_path = args.doc_splits + language + "/"
        data_process = DataPreprocessor(data_path_dir, splits_file_path, language)

        files_num = 0
        for split in ["train"]:
            events, non_events_sent, files_processed = data_process.extract_from_xml(split, language)
            files_num += files_processed

            total_sent = len(events) + len(non_events_sent)

            print("Number of files processed for language= ", language, " and split= ", split, " is= ", files_num,
                  "total number of sentences= ", total_sent, " and number of events= ", len(events))

            if args.method == "jmee":
                data_process.save_jmee_json(events, language, split, non_events_sent, pre_dir, args.use_neg_eg)
            else:
                print("Generating Trigger Detection only BIO Annotation ... ")
                data_process.get_triggers_bio(events, non_events_sent, language, split, pre_dir, args.use_neg_eg)
                print("Generating Argument Extraction only BIO Annotation ... ")
                data_process.get_arguments_bio(events, non_events_sent, language, split, pre_dir, args.use_neg_eg)
                print("Generating Joint BIO Annotation ... ")
                data_process.get_joint_dataset(events, non_events_sent, language, split, pre_dir, args.use_neg_eg)
