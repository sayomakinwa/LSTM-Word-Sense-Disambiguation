from nltk.corpus import wordnet as wn
from tensorflow.keras.models import *
import os
import json

from corpora import extract_eval_data
from my_utils import *


def predict_babelnet(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <BABELSynset>" format (e.g. "d000.s000.t000 bn:01234567n").
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    if (not resources_path.endswith("/")):
        resources_path = resources_path+"/"

    input_folder_path = input_path
    corpora_xml_path = input_path
    
    if (input_path.endswith(".xml")):
        input_folder_path = "/".join(input_path.split("/")[0:-1])

    if (not input_folder_path.endswith("/")):
        input_folder_path = input_folder_path+"/"

    if (os.path.isfile(output_path)):
        pred_file = output_path
        output_folder_path = "/".join(output_path.split("/")[0:-1])+"/"
    elif (os.path.isdir(output_path)):
        if (not output_path.endswith("/")):
            output_folder_path = output_path+"/"
        pred_file = output_folder_path+"pred_babelnet.txt"

    model_name = resources_path+"model.hdf5"
    print("LOADING RESOURCES...")
    model = load_model(model_name)

    #load the saved vocabularies
    with open(resources_path+"x_vocab.txt", 'r') as file:
        x_vocab = file.read()
    x_vocab = json.loads(x_vocab)

    with open(resources_path+"y_vocab.txt", 'r') as file:
        y_vocab = file.read()
    y_vocab = json.loads(y_vocab)
    id_to_words = {v:k for k, v in y_vocab.items()}

    bn2wn_mapping = load_bn2wn_mapping(resources_path+"babelnet2wordnet.tsv", True)

    #preparing the input data for prediction
    print("PREPARING EVALUATION DATA FOR PREDICTION...")
    extract_eval_data(corpora_xml_path, resources_path)

    sentences = load_test_dataset(input_folder_path+"sentences.txt")
    X_ = make_X(sentences, x_vocab)

    sentences_instances = load_sentence_instances(input_folder_path+"inst_temp_file.txt")

    #predicting and writing to file
    print("Predicting (line by line) and writing to file... This may take a little while...")
    k = 0
    inst_index = 0
    x_len = X_.shape
    with open(pred_file, "w") as file:
        for x in X_:
            if x.size != 0:
                x__ = np.expand_dims(x, axis=0)
                y_pred = model.predict(x__)

                # This loop is meant to handle one instance at a time, saved temporarily in the inst_temp_file.txt file,
                # loaded into the sentences_instances variable, till there's no instance left
                while True:
                    assoc_bn_synsets_vocab_pos = []
                    if inst_index not in sentences_instances:
                        break

                    inst = sentences_instances[inst_index]
                    if (int(inst[2]) != k):
                        break
                    else:
                        inst_index += 1
                        inst_pos_in_sent = int(inst[3])
                        inst_id = inst[1]

                        # Getting associated senses to the lemma of the instance selected
                        inst_synsets = wn.synsets(inst[0])
                        for wn_synset in inst_synsets:
                            wn_synset_id = "wn:" + str(wn_synset.offset()).zfill(8) + wn_synset.pos()
                            if wn_synset_id in bn2wn_mapping and bn2wn_mapping[wn_synset_id] in y_vocab:
                                assoc_bn_synsets_vocab_pos.append(y_vocab[bn2wn_mapping[wn_synset_id]])
                        
                        # Finding argmax over all associated synsets, and defaulting to MFS (pre saved to the vocab) where there's none
                        if assoc_bn_synsets_vocab_pos:
                            pred_word = y_pred[0, inst_pos_in_sent]
                            synset_probs = []
                            for pos in assoc_bn_synsets_vocab_pos:
                                synset_probs.append(pred_word[pos])

                            pred_sense = id_to_words[assoc_bn_synsets_vocab_pos[np.argmax(synset_probs)]]
                        else:
                            #MFS word = inst[0]
                            pred_sense = bn2wn_mapping[wn_mfs(inst[0])]

                        file.write("{} {}\n".format(inst_id, pred_sense))
            
            k = k+1
            if k % 100 < 1:
                print ("%d/%d lines done... A little moment more and everything will be done! :)" % (k,x_len[0]));

    del model, x_vocab, y_vocab, id_to_words, bn2wn_mapping, sentences, sentences_instances, X_, y_pred
    print("Prediction complete!")


def predict_wordnet_domains(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <wordnetDomain>" format (e.g. "d000.s000.t000 sport").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    if (not resources_path.endswith("/")):
        resources_path = resources_path+"/"

    input_folder_path = input_path
    corpora_xml_path = input_path
    
    if (input_path.endswith(".xml")):
        input_folder_path = "/".join(input_path.split("/")[0:-1])

    if (not input_folder_path.endswith("/")):
        input_folder_path = input_folder_path+"/"

    if (os.path.isfile(output_path)):
        pred_file = output_path
        output_folder_path = "/".join(output_path.split("/")[0:-1])+"/"
    elif (os.path.isdir(output_path)):
        if (not output_path.endswith("/")):
            output_folder_path = output_path+"/"
        pred_file = output_folder_path+"pred_domains.txt"

    model_name = resources_path+"model.hdf5"
    print("LOADING RESOURCES...")
    model = load_model(model_name)

    #load the saved vocabularies
    with open(resources_path+"x_vocab.txt", 'r') as file:
        x_vocab = file.read()
    x_vocab = json.loads(x_vocab)

    with open(resources_path+"y_vocab.txt", 'r') as file:
        y_vocab = file.read()
    y_vocab = json.loads(y_vocab)
    id_to_words = {v:k for k, v in y_vocab.items()}

    bn2wn_mapping = load_bn2wn_mapping(resources_path+"babelnet2wordnet.tsv", True)
    bn2dom_mapping = load_bn2wn_mapping(resources_path+"babelnet2wndomains.tsv")

    #preparing the input data for prediction
    print("PREPARING EVALUATION DATA FOR PREDICTION...")
    extract_eval_data(corpora_xml_path, resources_path)

    sentences = load_test_dataset(input_folder_path+"sentences.txt")
    X_ = make_X(sentences, x_vocab)

    sentences_instances = load_sentence_instances(input_folder_path+"inst_temp_file.txt")

    #predicting and writing to file
    print("Predicting (line by line) and writing to file... This may take a little while...")
    k = 0
    inst_index = 0
    x_len = X_.shape
    with open(pred_file, "w") as file:
        for x in X_:
            if x.size != 0:
                x__ = np.expand_dims(x, axis=0)
                y_pred = model.predict(x__)

                # This loop is meant to handle one instance at a time, saved temporarily in the inst_temp_file.txt file,
                # loaded into the sentences_instances variable, till there's no instance left
                while True:
                    assoc_bn_synsets_vocab_pos = []
                    if inst_index not in sentences_instances:
                        break

                    inst = sentences_instances[inst_index]
                    if (int(inst[2]) != k):
                        break
                    else:
                        inst_index += 1
                        inst_pos_in_sent = int(inst[3])
                        inst_id = inst[1]

                        # Getting associated senses to the lemma of the instance selected
                        inst_synsets = wn.synsets(inst[0])
                        for wn_synset in inst_synsets:
                            wn_synset_id = "wn:" + str(wn_synset.offset()).zfill(8) + wn_synset.pos()
                            if wn_synset_id in bn2wn_mapping and bn2wn_mapping[wn_synset_id] in y_vocab:
                                assoc_bn_synsets_vocab_pos.append(y_vocab[bn2wn_mapping[wn_synset_id]])
                        
                        # Finding argmax over all associated synsets, and defaulting to MFS (pre saved to the vocab) where there's none
                        if assoc_bn_synsets_vocab_pos:
                            pred_word = y_pred[0, inst_pos_in_sent]
                            synset_probs = []
                            for pos in assoc_bn_synsets_vocab_pos:
                                synset_probs.append(pred_word[pos])

                            pred_sense = id_to_words[assoc_bn_synsets_vocab_pos[np.argmax(synset_probs)]]
                            if pred_sense in bn2dom_mapping:
                                pred_dom = bn2dom_mapping[pred_sense]
                            else:
                                pred_dom = "factotum"
                        else:
                            #MFS word = inst[0]
                            pred_sense = bn2wn_mapping[wn_mfs(inst[0])]
                            if pred_sense in bn2dom_mapping:
                                pred_dom = bn2dom_mapping[pred_sense]
                            else:
                                pred_dom = "factotum"

                        file.write("{} {}\n".format(inst_id, pred_dom))
            
            k = k+1
            if k % 100 < 1:
                print ("%d/%d lines done... A little moment more and everything will be done! :)" % (k,x_len[0]));

    del model, x_vocab, y_vocab, id_to_words, bn2wn_mapping, sentences, sentences_instances, X_, y_pred
    print("Prediction complete!")


def predict_lexicographer(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <lexicographerId>" format (e.g. "d000.s000.t000 noun.animal").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    if (not resources_path.endswith("/")):
        resources_path = resources_path+"/"

    input_folder_path = input_path
    corpora_xml_path = input_path
    
    if (input_path.endswith(".xml")):
        input_folder_path = "/".join(input_path.split("/")[0:-1])

    if (not input_folder_path.endswith("/")):
        input_folder_path = input_folder_path+"/"

    if (os.path.isfile(output_path)):
        pred_file = output_path
        output_folder_path = "/".join(output_path.split("/")[0:-1])+"/"
    elif (os.path.isdir(output_path)):
        if (not output_path.endswith("/")):
            output_folder_path = output_path+"/"
        pred_file = output_folder_path+"pred_lex.txt"

    model_name = resources_path+"model.hdf5"
    print("LOADING RESOURCES...")
    model = load_model(model_name)

    #load the saved vocabularies
    with open(resources_path+"x_vocab.txt", 'r') as file:
        x_vocab = file.read()
    x_vocab = json.loads(x_vocab)

    with open(resources_path+"y_vocab.txt", 'r') as file:
        y_vocab = file.read()
    y_vocab = json.loads(y_vocab)
    id_to_words = {v:k for k, v in y_vocab.items()}

    bn2wn_mapping = load_bn2wn_mapping(resources_path+"babelnet2wordnet.tsv", True)
    bn2lex_mapping = load_bn2wn_mapping(resources_path+"babelnet2lexnames.tsv")

    #preparing the input data for prediction
    print("PREPARING EVALUATION DATA FOR PREDICTION...")
    extract_eval_data(corpora_xml_path, resources_path)

    sentences = load_test_dataset(input_folder_path+"sentences.txt")
    X_ = make_X(sentences, x_vocab)

    sentences_instances = load_sentence_instances(input_folder_path+"inst_temp_file.txt")

    #predicting and writing to file
    print("Predicting (line by line) and writing to file... This may take a little while...")
    k = 0
    inst_index = 0
    x_len = X_.shape
    with open(pred_file, "w") as file:
        for x in X_:
            if x.size != 0:
                x__ = np.expand_dims(x, axis=0)
                y_pred = model.predict(x__)

                # This loop is meant to handle one instance at a time, saved temporarily in the inst_temp_file.txt file,
                # loaded into the sentences_instances variable, till there's no instance left
                while True:
                    assoc_bn_synsets_vocab_pos = []
                    if inst_index not in sentences_instances:
                        break

                    inst = sentences_instances[inst_index]
                    if (int(inst[2]) != k):
                        break
                    else:
                        inst_index += 1
                        inst_pos_in_sent = int(inst[3])
                        inst_id = inst[1]

                        # Getting associated senses to the lemma of the instance selected
                        inst_synsets = wn.synsets(inst[0])
                        for wn_synset in inst_synsets:
                            wn_synset_id = "wn:" + str(wn_synset.offset()).zfill(8) + wn_synset.pos()
                            if wn_synset_id in bn2wn_mapping and bn2wn_mapping[wn_synset_id] in y_vocab:
                                assoc_bn_synsets_vocab_pos.append(y_vocab[bn2wn_mapping[wn_synset_id]])
                        
                        # Finding argmax over all associated synsets, and defaulting to MFS (pre saved to the vocab) where there's none
                        if assoc_bn_synsets_vocab_pos:
                            pred_word = y_pred[0, inst_pos_in_sent]
                            synset_probs = []
                            for pos in assoc_bn_synsets_vocab_pos:
                                synset_probs.append(pred_word[pos])

                            pred_sense = id_to_words[assoc_bn_synsets_vocab_pos[np.argmax(synset_probs)]]
                            if pred_sense in bn2lex_mapping:
                                pred_lex = bn2lex_mapping[pred_sense]
                            else:
                                pred_lex = "adj.all"
                        else:
                            #MFS word = inst[0]
                            pred_sense = bn2wn_mapping[wn_mfs(inst[0])]
                            if pred_sense in bn2lex_mapping:
                                pred_lex = bn2lex_mapping[pred_sense]
                            else:
                                pred_lex = "adj.all"

                        file.write("{} {}\n".format(inst_id, pred_lex))
            
            k = k+1
            if k % 100 < 1:
                print ("%d/%d lines done... A little moment more and everything will be done! :)" % (k,x_len[0]));

    del model, x_vocab, y_vocab, id_to_words, bn2wn_mapping, sentences, sentences_instances, X_, y_pred
    print("Prediction complete!")