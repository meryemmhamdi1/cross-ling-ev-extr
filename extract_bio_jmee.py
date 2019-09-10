from DataModule.data_utils import *
from DataModule.ace_utils import *
from get_args import *
import json

import os


def pretty_str(a):
    a = a.upper()
    if a == 'O':
        return a
    elif a[1] == '-':
        return a[:2] + "|".join(a[2:].split("-")).replace(":", "||")
    else:
        return "|".join(a.split("-")).replace(":", "||")


if __name__ == '__main__':

    args = get_arguments()

    for language in args.languages.split(","):

        if not os.path.exists(args.pre_dir):
            os.makedirs(args.pre_dir)

        data_path_dir = "../"+args.root_dir + args.pre_dir + "JMEE_Dataset/"
        data_pre = "../"+args.root_dir + args.pre_dir + "tagging_new/TriggerIdentification/" + language

        for lang in languages:
            for split in ["train", "dev", "test"]:
                with open(data_path_dir+lang+"/JMEE_"+split+".json") as file:
                    data = json.load(file)

                word_list = []
                trigger_list = []
                for sent in data:
                    triggerLabel = ["O" for _ in range(len(sent["words"]))]

                    def assignTriggerLabel(index, label):
                        triggerLabel[index] = label

                    for eventJson in sent["golden-event-mentions"]:
                        triggerJson = eventJson["trigger"]
                        start = triggerJson["start"]
                        end = triggerJson["end"]
                        etype = eventJson["event_type"]
                        assignTriggerLabel(start, "B-" + etype)
                        for i in range(start + 1, end):
                            assignTriggerLabel(i, "I-" + etype)

                    word_list.append(sent["words"])
                    trigger_list.append(triggerLabel)

                with open(data_pre+split+".txt", "w") as file:
                    for i, sent in enumerate(word_list):
                        for j, word in enumerate(sent):
                            file.write(word + "\t" + trigger_list[i][j] + "\n")
                        file.write("\n")
