from my_utils import *
from nltk.tokenize import RegexpTokenizer

def score_predict1(prediction, gold_file, bn2wn_mapping_file):
    
    gold_mapping = load_gold_data(gold_file)
    pred_mapping = load_gold_data(prediction)

    bn2wn_mapping = load_bn2wn_mapping(bn2wn_mapping_file, True)

    count = len(gold_mapping)
    rights = 0
    k=0
    not_found = 0

    for inst_id, inst_sense_key in gold_mapping.items():
        k+1
        #print("Scoring {:,d}/{:,d} instances...".format(k, count), end="\r")
        if inst_id in pred_mapping:
            wn_synset = wn.lemma_from_key(inst_sense_key).synset()
            wn_synset_id = "wn:" + str(wn_synset.offset()).zfill(8) + wn_synset.pos()
            if (pred_mapping[inst_id] == bn2wn_mapping[wn_synset_id]):
                rights += 1
        else:
            not_found += 1

    #print("\nCorrect predictions: {}".format(rights))
    #print("F1 score: {}".format(rights/count))
    #print("No of instances with no prediction: {}".format(not_found))

    return rights/count


def score_predict_lex(prediction, gold_file, resources_path):
    
    gold_mapping = load_gold_data(gold_file)
    pred_mapping = load_gold_data(prediction)

    if (not resources_path.endswith("/")):
        resources_path = resources_path+"/"

    bn2wn_mapping = load_bn2wn_mapping(resources_path+"babelnet2wordnet.tsv", True)
    bn2lex_mapping = load_bn2wn_mapping(resources_path+"babelnet2lexnames.tsv")

    count = len(gold_mapping)
    rights = 0
    k=0
    not_found = 0

    for inst_id, inst_sense_key in gold_mapping.items():
        k+1
        #print("Scoring {:,d}/{:,d} instances...".format(k, count), end="\r")
        if inst_id in pred_mapping:
            wn_synset = wn.lemma_from_key(inst_sense_key).synset()
            wn_synset_id = "wn:" + str(wn_synset.offset()).zfill(8) + wn_synset.pos()

            if bn2wn_mapping[wn_synset_id] in bn2lex_mapping:
                lex = bn2lex_mapping[bn2wn_mapping[wn_synset_id]]
            else:
                lex = "adj.all"

            if (pred_mapping[inst_id] == lex):
                rights += 1
        else:
            not_found += 1

    #print("\nCorrect predictions: {}".format(rights))
    #print("F1 score: {}".format(rights/count))
    #print("No of instances with no prediction: {}".format(not_found))

    return rights/count


def score_predict_dom(prediction, gold_file, resources_path):
    
    gold_mapping = load_gold_data(gold_file)
    pred_mapping = load_gold_data(prediction)

    if (not resources_path.endswith("/")):
        resources_path = resources_path+"/"

    bn2wn_mapping = load_bn2wn_mapping(resources_path+"babelnet2wordnet.tsv", True)
    bn2dom_mapping = load_bn2wn_mapping(resources_path+"babelnet2wndomains.tsv")

    count = len(gold_mapping)
    rights = 0
    k=0
    not_found = 0

    for inst_id, inst_sense_key in gold_mapping.items():
        k+1
        #print("Scoring {:,d}/{:,d} instances...".format(k, count), end="\r")
        if inst_id in pred_mapping:
            wn_synset = wn.lemma_from_key(inst_sense_key).synset()
            wn_synset_id = "wn:" + str(wn_synset.offset()).zfill(8) + wn_synset.pos()

            if bn2wn_mapping[wn_synset_id] in bn2dom_mapping:
                dom = bn2dom_mapping[bn2wn_mapping[wn_synset_id]]
            else:
                dom = "factotum"

            if (pred_mapping[inst_id] == dom):
                rights += 1
        else:
            not_found += 1

    #print("\nCorrect predictions: {}".format(rights))
    #print("F1 score: {}".format(rights/count))
    #print("No of instances with no prediction: {}".format(not_found))

    return rights/count