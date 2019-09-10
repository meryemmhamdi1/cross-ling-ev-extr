import argparse


def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--languages", "-lang", type=str,  default="English")
    parser.add_argument("--root-dir", "-rd", type=str, help="Root path of directory for LDC2006T06 ACE dataset release")
    parser.add_argument("--data-ace-path", "-dap", type=str,
                        default="EventsExtraction/ACE/Raw/ACE2005-TrainingData-V6.0/")
    parser.add_argument("--pre-dir", "-pd", type=str, default="EventsExtraction/ACE/Preprocessed/")
    parser.add_argument("--w2v-dir", "-w2v", type=str, default="")
    parser.add_argument("--method", "-mth", type=str, default="jmee", help="jmee or tagging")
    parser.add_argument("--doc-splits", "-dspl", type=str, default="DataModule/doc_splits/")
    parser.add_argument("--use-neg-eg", "-une", type=bool, default=False)
    parser.add_argument("--jmee-splits", "-js", type=str, default="doc_splits/")
    parser.add_argument("--split-option", "-so", type=str, default="jmee",
                        help="- cv for cross-validation"
                             "- jmee for jmee-splits"
                             "- randoc for another doc split" 
                             "- ransent for another sent split")

    parser.add_argument("--json-dir", "-jdir", type=str,  default="",
                        help="directory for json data with data extracted from APF Annotation")

    parser.add_argument("--train-prop", "-tr-pr", type=float, default=0.88)
    parser.add_argument("--test-prop", "-te-pr", type=float, default=0.07)
    parser.add_argument("--dev-prop", "-de-pr", type=float, default=0.05)


    return parser.parse_args()
