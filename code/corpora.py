from lxml import etree
from nltk.corpus import wordnet as wn
from my_utils import *
from nltk.tokenize import RegexpTokenizer

def extract_training_data(corpora_xml_path: str, gold_mapping_path: str, resources_path: str, outfolder_path: str) -> None:
    """
    :param corpora_path; Full path to the corpora_path to be parsed
    :param resources_path; Full path to the BabelNet to WordNet file mapping for filtering the corpora
    :param outfile_path; Folder path to write the sentences extracted from the corpora
    :param c_type; Corpora type "precision" or "coverage"
    :return None

    THIS FUNCTION HANDLES ONLY EUROSENSE CORPORA
    """
    print("Loading gold data...")
    gold_mapping = load_gold_data(gold_mapping_path)

    print("Loading mapping files...")
    if (not resources_path.endswith("/")):
        resources_path = resources_path+"/"

    bn2wn_mapping = load_bn2wn_mapping(resources_path+"babelnet2wordnet.tsv", True)
    bn2dom_mapping = load_bn2wn_mapping(resources_path+"babelnet2wndomains.tsv", False)
    bn2lex_mapping = load_bn2wn_mapping(resources_path+"babelnet2lexnames.tsv", False)

    sentence_x = ""
    sentence_y = ""
    sentence_lex = ""
    sentence_dom = ""
    sentence_pos = ""

    context = etree.iterparse(corpora_xml_path, events=('start', 'end'))
    tokenizer = RegexpTokenizer(r'\w+') #using this to get rid of punctuation marks

    sentence_count = 0

    xpath = outfolder_path + "/trainX.txt"
    ypath = outfolder_path + "/trainy.txt"
    ypath_lex = outfolder_path + "/trainy_lex.txt"
    ypath_dom = outfolder_path + "/trainy_dom.txt"
    ypath_pos = outfolder_path + "/trainy_pos.txt"
    
    flag_wf = False
    flag_inst = False

    with open(xpath, 'w') as x_file:
        with open(ypath, 'w') as y_file:
            with open(ypath_lex, 'w') as lex_file:
                with open(ypath_dom, 'w') as dom_file:
                    with open(ypath_pos, 'w') as pos_file:

                        for event, elem in context:
                            if (event == 'start'):
                                if (elem.tag == 'wf'):
                                    lemma = elem.attrib['lemma']
                                    if (elem.text):
                                        text = elem.text

                                        if (len(lemma) > 1):
                                            lemma = lemma.replace("-","_")

                                        if (len(text) > 1):
                                            text = text.replace("-","_")

                                        if (len(text.split()) > 1):
                                            text = text.replace(" ","_")

                                        text = tokenizer.tokenize(text)
                                        lemma = tokenizer.tokenize(lemma)

                                        if (text):
                                            sentence_pos = sentence_pos + " " + "pos:"+elem.attrib['pos']
                                            if (lemma):
                                                sentence_x = sentence_x + " " + "".join(text)
                                                sentence_y = sentence_y + " " + "".join(lemma)
                                                sentence_dom = sentence_dom + " " + "".join(lemma)
                                                sentence_lex = sentence_lex + " " + "".join(lemma)
                                            else:
                                                sentence_x = sentence_x + "".join(text)
                                                sentence_y = sentence_y + "".join(lemma)
                                                sentence_dom = sentence_dom + "".join(lemma)
                                                sentence_lex = sentence_lex + "".join(lemma)
                                    
                                    else :
                                        flag_wf = True
                                    
                                elif (elem.tag == 'instance'):
                                    sentence_pos = sentence_pos + " " + "pos:"+elem.attrib['pos']

                                    sense_key = gold_mapping[elem.attrib['id']]
                                    wn_synset = wn.lemma_from_key(sense_key).synset()
                                    wn_synset_id = "wn:" + str(wn_synset.offset()).zfill(8) + wn_synset.pos()

                                    sentence_y = sentence_y + " " + bn2wn_mapping[wn_synset_id]
                                    if bn2wn_mapping[wn_synset_id] in bn2dom_mapping:
                                        sentence_dom = sentence_dom + " " + "dom:"+bn2dom_mapping[bn2wn_mapping[wn_synset_id]]
                                    else:
                                        #sentence_dom = sentence_dom + " " + "dom:factotum"
                                        sentence_dom = sentence_dom + " " + "NAN"
                                    
                                    if bn2wn_mapping[wn_synset_id] in bn2lex_mapping:
                                        sentence_lex = sentence_lex + " " + "lex:"+bn2lex_mapping[bn2wn_mapping[wn_synset_id]]
                                    else:
                                        #sentence_lex = sentence_lex + " " + "lex:adj.all"
                                        sentence_lex = sentence_lex + " " + "NAN"

                                    if (elem.text):
                                        text = elem.text
                                        if (len(text) > 1):
                                            text = text.replace("-","_")

                                        if (len(text.split()) > 1):
                                            text = text.replace(" ","_")

                                        sentence_x = sentence_x + " " + text
                                    else :
                                        flag_inst = True

                            elif (event == 'end'):
                                if (flag_inst):
                                    text = elem.text
                                    if (len(text) > 1):
                                        text = text.replace("-","_")

                                    if (len(text.split()) > 1):
                                        text = text.replace(" ","_")

                                    sentence_x = sentence_x + " " + text
                                    flag_inst = False

                                if (flag_wf):
                                    text = elem.text
                                    if (len(text) > 1):
                                        text = text.replace("-","_")

                                    if (len(text.split()) > 1):
                                        text = text.replace(" ","_")

                                    text = tokenizer.tokenize(text)
                                    if (text):
                                        sentence_x = sentence_x + " " + "".join(text)
                                        sentence_y = sentence_y + " " + "".join(lemma)
                                        sentence_dom = sentence_dom + "".join(lemma)
                                        sentence_lex = sentence_lex + "".join(lemma)

                                    flag_wf = False

                                if (elem.tag == 'sentence'):
                                    if (sentence_x):
                                        x_file.write("{}\n".format(sentence_x[1:]))
                                        y_file.write("{}\n".format(sentence_y[1:]))
                                        lex_file.write("{}\n".format(sentence_lex[1:]))
                                        dom_file.write("{}\n".format(sentence_dom[1:]))
                                        pos_file.write("{}\n".format(sentence_pos[1:]))
                                        sentence_count += 1
                                        print("{:,d} sentences extracted...".format(sentence_count), end="\r")
                                        
                                        sentence_x = ""
                                        sentence_y = ""
                                        sentence_lex = ""
                                        sentence_dom = ""
                                        sentence_pos = ""
                                    
                            #Freeing up memory before parsing the next tag
                            elem.clear()
                            while elem.getprevious() is not None:
                                del elem.getparent()[0]
    print ("\nDone!")



def extract_eval_data(corpora_xml_path: str, resources_path: str) -> None:
    """
    :param corpora_path; Full path to the corpora_path to be parsed
    :param resources_path; Full path to the BabelNet to WordNet file mapping for filtering the corpora
    :param outfile_path; Folder path to write the sentences extracted from the corpora
    :param c_type; Corpora type "precision" or "coverage"
    :return None

    THIS FUNCTION HANDLES ONLY EUROSENSE CORPORA
    """
    input_folder_path = "/".join(corpora_xml_path.split("/")[:-1])+"/"

    sentence_x = ""
    instances = []
    sentence_pos = -1
    context = etree.iterparse(corpora_xml_path, events=('start', 'end'))
    tokenizer = RegexpTokenizer(r'\w+')

    sentence_count = 0
    
    xpath = input_folder_path + "sentences.txt"
    inst_path = input_folder_path + "inst_temp_file.txt"
    
    flag_wf = False
    flag_inst = False

    with open(inst_path, 'w') as inst_file:
        with open(xpath, 'w') as x_file:
            for event, elem in context:
                if (event == 'start'):
                    if (elem.tag == 'wf'):
                        if (elem.text):
                            text = elem.text
                            if (len(text) > 1):
                                text = text.replace("-","_")

                            if (len(text.split()) > 1):
                                text = text.replace(" ","_")

                            text = tokenizer.tokenize(elem.text)
                            if (text):
                                sentence_pos += 1
                                sentence_x = sentence_x + " " + "".join(text)
                        
                        else :
                            flag_wf = True
                        
                    elif (elem.tag == 'instance'):
                        ##########################################
                        sentence_pos += 1
                        instances.append([elem.attrib['lemma'], elem.attrib['id'], sentence_count, sentence_pos])
                        ##########################################

                        if (elem.text):
                            text = elem.text
                            if (len(text) > 1):
                                text = text.replace("-","_")

                            if (len(text.split()) > 1):
                                text = text.replace(" ","_")

                            sentence_x = sentence_x + " " + text
                        else :
                            flag_inst = True

                elif (event == 'end'):
                    if (flag_inst):
                        text = elem.text
                        if (len(text) > 1):
                            text = text.replace("-","_")

                        if (len(text.split()) > 1):
                            text = text.replace(" ","_")

                        sentence_x = sentence_x + " " + text
                        flag_inst = False

                    if (flag_wf):
                        text = elem.text
                        if (len(text) > 1):
                            text = text.replace("-","_")

                        if (len(text.split()) > 1):
                                text = text.replace(" ","_")

                        text = tokenizer.tokenize(elem.text)
                        if (text):
                            sentence_pos += 1
                            sentence_x = sentence_x + " " + "".join(text)
                        flag_wf = False

                    if (elem.tag == 'sentence'):
                        if (sentence_x):
                            x_file.write("{}\n".format(sentence_x[1:]))
                            
                            for instance in instances:
                                inst_file.write("{}\t{}\t{}\t{}\n".format(instance[0], instance[1], instance[2], instance[3]))

                            sentence_count += 1
                            sentence_pos = -1
                            print("{:,d} sentences extracted...".format(sentence_count), end="\r")
                            
                            sentence_x = ""
                            instances = []
                        
                #Freeing up memory before parsing the next tag
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
    print ("\nEvaluation data extraction completed")