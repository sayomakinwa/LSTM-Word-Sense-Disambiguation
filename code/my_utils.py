from typing import Dict, List, Tuple
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from string import punctuation as punct
import numpy as np
from nltk.corpus import wordnet as wn
#from predict import *


class InputSentences(object):
    def __init__(self, inputpath):
        self.inputpath = inputpath
 
    def __iter__(self):
        if (os.path.isdir(self.inputpath)):
            for filename in os.listdir(self.inputpath):
                with open(os.path.join(self.inputpath, filename), 'r') as handle:
                    for line in handle:
                        # the [2:-1] indexing was so as to remove the leading "b'" and trailing "'" for each line
                        # using yield so the output can be a generator, optimizing the use of memory
                        yield line.strip().split()
                        
        elif (os.path.isfile(self.inputpath)):        
            with open(self.inputpath, 'r') as handle:
                for line in handle:
                    # the [2:-1] indexing was so as to remove the leading "b'" and trailing "'" for each line
                    # using yield so the output can be a generator, optimizing the use of memory
                    yield line.strip()[2:-1].split()



def load_bn2wn_mapping(bn2wn_mapping_path: str, flip_dxn = False) -> Dict[str, str]:
    """
    :param bn2wn_mapping_path; Full path to the BabelNet to WordNet file mapping for filtering the corpora
    :return bn2wn_mapping; the BabelNet to WordNet mapping file encoded as a dictionary with the BabelNet IDs as keys
    """
    first = 0
    second = 1
    if (flip_dxn):
        first = 1
        second = 0
    bn2wn_mapping = dict()
    with open(bn2wn_mapping_path, 'r') as handle:
        for line in handle:
            line = line.strip().split("\t")
            if (line):
                bn2wn_mapping[line[first]] = line[second]

    return bn2wn_mapping


def load_gold_data(gold_data_path: str) -> Dict[str, str]:
    gold_mapping = dict()
    with open(gold_data_path, 'r') as handle:
        for line in handle:
            line = line.strip().split(" ")
            if (line):
                gold_mapping[line[0]] = line[1]

    return gold_mapping


def load_test_dataset(file_path: str) -> List[str]:
    """
    :param file_path; Path to the input dataset
    :return sentences; List of sentences in input_file
    """
    sentences = []
    #with open(file_path, "r", encoding="utf-8-sig") as file:
    with open(file_path, "r") as file:
        for line in file:
            sentences.append(line.strip())

    return sentences


def load_sentence_instances(inst_file_path: str) -> Dict[str, str]:
    instances = dict()
    k=0
    with open(inst_file_path, 'r') as handle:
        for line in handle:
            line = line.strip().split("\t")
            if (line):
                #instances[k] = [line[2], line[0], line[3], line[1]]
                instances[k] = [line[0], line[1], line[2], line[3]]
                k += 1

    return instances


def make_X(sentences: List[str], vocab: Dict[str, int]) -> np.ndarray:
    """
    :param sentences; List of sentences
    :param unigrams_vocab; Unigram vocabulary
    :param bigrams_vocab; Bigram vocabulary
    :return X; Matrix storing all sentences' feature vector 
    """
    X1 = []
    for sentence in sentences:
        x_temp = []
        for word in sentence.split():
            if word in vocab:
                x_temp.append(vocab[word])
            else:
                x_temp.append(vocab["UNK"])

        X1.append(np.array(x_temp))

    X1 = np.array(X1)
    return X1


def wn_mfs(word):
    synsets = wn.synsets(word)
    sense2freq = {}
    mfs = ""
    mfs_freq = 0

    for s in synsets:
        freq = 0  
        for lemma in s.lemmas():
            freq += lemma.count()

        if freq > mfs_freq or mfs_freq == 0:
            mfs = "wn:" + str(s.offset()).zfill(8) + s.pos()
            mfs_freq = freq
        
    return mfs



# def run_tests1(score_file):
    
#     resources_path = '../resources'
#     bn2wn_mapping_file = '../resources/babelnet2wordnet.tsv'
    
#     all_scores = []
#     if (os.path.isfile(score_file)):
#         with open(score_file, 'r') as file:
#             for line in file:
#                 all_scores.append(line.split(","))
        
#     print("PREDICTING FOR SE2...")
#     se2_scores = ['SE2']

#     input_path = '../data/Evaluation_Datasets/senseval2/senseval2.data.xml'
#     output_path = '../data/Evaluation_Datasets/senseval2'
#     gold_file =  '../data/Evaluation_Datasets/senseval2/senseval2.gold.key.txt'

#     pred_babelnet = '../data/Evaluation_Datasets/senseval2/pred_babelnet.txt'
#     pred_lex = '../data/Evaluation_Datasets/senseval2/pred_lex.txt'
#     pred_domains = '../data/Evaluation_Datasets/senseval2/pred_domains.txt'

#     predict_babelnet(input_path, output_path, resources_path)
#     score = score_predict1(pred_babelnet, gold_file, bn2wn_mapping_file)
#     se2_scores.append(score*100)

#     predict_lexicographer(input_path, output_path, resources_path)
#     score = score_predict_lex(pred_lex, gold_file, resources_path)
#     se2_scores.append(score*100)

#     predict_wordnet_domains(input_path, output_path, resources_path)
#     score = score_predict_dom(pred_domains, gold_file, resources_path)
#     se2_scores.append(score*100)

#     all_scores.append(se2_scores)
    

#     #########################################################

#     print("\n\nPREDICTING FOR SE3...")
#     se3_scores = ['SE3_(Devs)']

#     input_path = '../data/Evaluation_Datasets/senseval3/senseval3.data.xml'
#     output_path = '../data/Evaluation_Datasets/senseval3'
#     gold_file =  '../data/Evaluation_Datasets/senseval3/senseval3.gold.key.txt'

#     pred_babelnet = '../data/Evaluation_Datasets/senseval3/pred_babelnet.txt'
#     pred_lex = '../data/Evaluation_Datasets/senseval3/pred_lex.txt'
#     pred_domains = '../data/Evaluation_Datasets/senseval3/pred_domains.txt'

#     predict_babelnet(input_path, output_path, resources_path)
#     score = score_predict1(pred_babelnet, gold_file, bn2wn_mapping_file)
#     se3_scores.append(score*100)

#     predict_lexicographer(input_path, output_path, resources_path)
#     score = score_predict_lex(pred_lex, gold_file, resources_path)
#     se3_scores.append(score*100)

#     predict_wordnet_domains(input_path, output_path, resources_path)
#     score = score_predict_dom(pred_domains, gold_file, resources_path)
#     se3_scores.append(score*100)

#     all_scores.append(se3_scores)
    
    
#     ###########################################################
        
#     with open(score_file, "w") as file:
#         for scores in all_scores:
#             file.write("{},{:.1f},{:.1f},{:.1f}\n".format(scores[0], float(scores[1]), float(scores[2]), float(scores[3])))



# def run_tests2(score_file):
    
#     resources_path = '../resources'
#     bn2wn_mapping_file = '../resources/babelnet2wordnet.tsv'
    
#     all_scores = []
#     if (os.path.isfile(score_file)):
#         with open(score_file, 'r') as file:
#             for line in file:
#                 all_scores.append(line.split(","))
                
    
#     print("\n\nPREDICTING FOR SE07...")
#     se07_scores = ['SE07']
    
#     input_path = '../data/Evaluation_Datasets/semeval2007/semeval2007.data.xml'
#     output_path = '../data/Evaluation_Datasets/semeval2007'
#     gold_file =  '../data/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt'
    
#     pred_babelnet = '../data/Evaluation_Datasets/semeval2007/pred_babelnet.txt'
#     pred_lex = '../data/Evaluation_Datasets/semeval2007/pred_lex.txt'
#     pred_domains = '../data/Evaluation_Datasets/semeval2007/pred_domains.txt'

#     predict_babelnet(input_path, output_path, resources_path)
#     score = score_predict1(pred_babelnet, gold_file, bn2wn_mapping_file)
#     se07_scores.append(score*100)
    
#     predict_lexicographer(input_path, output_path, resources_path)
#     score = score_predict_lex(pred_lex, gold_file, resources_path)
#     se07_scores.append(score*100)
    
#     predict_wordnet_domains(input_path, output_path, resources_path)
#     score = score_predict_dom(pred_domains, gold_file, resources_path)
#     se07_scores.append(score*100)
    
#     all_scores.append(se07_scores)
    
#     #########################################################
    
#     print("\n\nPREDICTING FOR SE13...")
#     se13_scores = ['SE13']
    
#     input_path = '../data/Evaluation_Datasets/semeval2013/semeval2013.data.xml'
#     output_path = '../data/Evaluation_Datasets/semeval2013'
#     gold_file =  '../data/Evaluation_Datasets/semeval2013/semeval2013.gold.key.txt'
    
#     pred_babelnet = '../data/Evaluation_Datasets/semeval2013/pred_babelnet.txt'
#     pred_lex = '../data/Evaluation_Datasets/semeval2013/pred_lex.txt'
#     pred_domains = '../data/Evaluation_Datasets/semeval2013/pred_domains.txt'

#     predict_babelnet(input_path, output_path, resources_path)
#     score = score_predict1(pred_babelnet, gold_file, bn2wn_mapping_file)
#     se13_scores.append(score*100)
    
#     predict_lexicographer(input_path, output_path, resources_path)
#     score = score_predict_lex(pred_lex, gold_file, resources_path)
#     se13_scores.append(score*100)
    
#     predict_wordnet_domains(input_path, output_path, resources_path)
#     score = score_predict_dom(pred_domains, gold_file, resources_path)
#     se13_scores.append(score*100)
    
#     all_scores.append(se13_scores)
    
#     #########################################################
    
    
#     with open(score_file, "w") as file:
#         for scores in all_scores:
#             file.write("{},{:.1f},{:.1f},{:.1f}\n".format(scores[0], float(scores[1]), float(scores[2]), float(scores[3])))
            
            
            
    
    

# def run_tests3(score_file):
    
#     resources_path = '../resources'
#     bn2wn_mapping_file = '../resources/babelnet2wordnet.tsv'
    
#     all_scores = []
#     if (os.path.isfile(score_file)):
#         with open(score_file, 'r') as file:
#             for line in file:
#                 all_scores.append(line.split(","))
                
    
#     print("\n\nPREDICTING FOR SE15...")
#     se15_scores = ['SE15']
    
#     input_path = '../data/Evaluation_Datasets/semeval2015/semeval2015.data.xml'
#     output_path = '../data/Evaluation_Datasets/semeval2015'
#     gold_file =  '../data/Evaluation_Datasets/semeval2015/semeval2015.gold.key.txt'
    
#     pred_babelnet = '../data/Evaluation_Datasets/semeval2015/pred_babelnet.txt'
#     pred_lex = '../data/Evaluation_Datasets/semeval2015/pred_lex.txt'
#     pred_domains = '../data/Evaluation_Datasets/semeval2015/pred_domains.txt'

#     predict_babelnet(input_path, output_path, resources_path)
#     score = score_predict1(pred_babelnet, gold_file, bn2wn_mapping_file)
#     se15_scores.append(score*100)
    
#     predict_lexicographer(input_path, output_path, resources_path)
#     score = score_predict_lex(pred_lex, gold_file, resources_path)
#     se15_scores.append(score*100)
    
#     predict_wordnet_domains(input_path, output_path, resources_path)
#     score = score_predict_dom(pred_domains, gold_file, resources_path)
#     se15_scores.append(score*100)
    
#     all_scores.append(se15_scores)
    
    
#     #####################################################################
    
    
#     print("\n\nPREDICTING FOR ALL...")
#     se_ALL_scores = ['ALL']
    
#     input_path = '../data/Evaluation_Datasets/ALL/ALL.data.xml'
#     output_path = '../data/Evaluation_Datasets/ALL'
#     gold_file =  '../data/Evaluation_Datasets/ALL/ALL.gold.key.txt'
    
#     pred_babelnet = '../data/Evaluation_Datasets/ALL/pred_babelnet.txt'
#     pred_lex = '../data/Evaluation_Datasets/ALL/pred_lex.txt'
#     pred_domains = '../data/Evaluation_Datasets/ALL/pred_domains.txt'

#     predict_babelnet(input_path, output_path, resources_path)
#     score = score_predict1(pred_babelnet, gold_file, bn2wn_mapping_file)
#     se_ALL_scores.append(score*100)
    
#     predict_lexicographer(input_path, output_path, resources_path)
#     score = score_predict_lex(pred_lex, gold_file, resources_path)
#     se_ALL_scores.append(score*100)
    
#     predict_wordnet_domains(input_path, output_path, resources_path)
#     score = score_predict_dom(pred_domains, gold_file, resources_path)
#     se_ALL_scores.append(score*100)
    
#     all_scores.append(se_ALL_scores)
    
    
#     with open(score_file, "w") as file:
#         for scores in all_scores:
#             file.write("{},{:.1f},{:.1f},{:.1f}\n".format(scores[0], float(scores[1]), float(scores[2]), float(scores[3])))

# def replace_words(sentence: str, anchors: List[str], lemmas: List[str], bn_keys: List[str]) -> str:
#     """
#     :param sentence; An extracted sentence from the corpora
#     :param anchors; A list of anchor words in the sentence to be replaced
#     :param lemmas; A list of corresponding lemmas with matching position in anchors
#     :param bn_keys; A list of corresponding BabelNet synset ID's with matching position in anchors
#     :return new_sentence; The resulting sentence from the replacements
#     """
#     new_sentence = ""
#     sentence = sentence.lower().split()
#     for word in sentence:
#         for k in range(len(anchors)):
#             if (word == anchors[k] or anchors[k] in word.split("-")):
#                 word = lemmas[k]+"_"+bn_keys[k]

#         new_sentence = new_sentence + word + " "

#     return new_sentence[:-1]



# def remove_stopwords(input_path: str, output_path: str, lang: str) -> None:
#     """
#     :param input_path; Path to the already parsed file
#     :param output_path; Path to write the new file
#     :param lang; 'english', 'italian', etc, for which to remove stopwords for
#     """
#     eng_stopwords = set(stopwords.words(lang))
#     sentence_count = 0
#     with open(input_path, 'r') as handle1:
#         with open(output_path, 'w') as handle2:
#             for line in handle1:
#                 words = line.strip()[2:-1].split()
#                 new_words = []
#                 for word in words:
#                     if word not in eng_stopwords and word not in punct:
#                         new_words.append(word)

#                 handle2.write("{}\n".format(" ".join(new_words)))
#                 sentence_count += 1
#                 print("{:,d} sentences extracted...".format(sentence_count), end="\r")



# def save_embeddings(path: str, model_wv: KeyedVectors) -> None:
#     """
#     :param path; Path to save the sense embeddings to
#     :param model_wv; The trained model from which to extract the sense embeddings

#     I decided to use two loops here because I wanted to be able to write the length of the senses first
#     """
#     sense_vocab = dict()
#     for word in model_wv.vocab.keys():
#         if "_bn:" in word:
#             sense_vocab[word] = model_wv[word]
            
#     with open(path, "w") as handle:
#         handle.write("{} {}\n".format(len(sense_vocab), model_wv.vector_size))
#         for sense, vector in sense_vocab.items():
#             handle.write("{} {}\n".format(sense, " ".join(map(str, list(vector)))))



# def plot_helper(xvalues: List, yvalues: List, xlabel: str = "", ylabel: List[str] = ["",""]) -> None:
#     """
#     :param xvalues; Path to the already parsed file
#     :param yvalues; Path to write the new file
#     :param xlabel; Label for the x-axis
#     :param ylabel; A list of two labels for the y axes

#     NOTE: the x axes both have the same label, while the y axes have different labels since it's a graph of correlations and p-values together
#     """
#     fig = plt.gcf()
#     fig.set_size_inches(8.5, 3)

#     plt.subplot(1, 2, 1)
#     plt.plot(xvalues, yvalues[0], 'b')
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel[0])
#     plt.title("Correlation graph")
#     plt.grid()

#     plt.subplot(1, 2, 2)
#     plt.plot(xvalues, yvalues[1], 'r')
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel[1])
#     plt.title("p-value graph")
#     plt.grid()

#     plt.tight_layout()
#     plt.show()



# def get_first_similar_words(word: str, w2v_model:KeyedVectors, count: int = 5) -> Tuple:
#     """
#     :param word; The word for which associated sense embeddings is to be gotten
#     :param w2v_model; Model of the sense embeddings to check for associated senses
#     :param count; The maximum number of sense embeddings to return
#     :return senses; A list of the associated senses
#     :return embeddings; An np array of corressponding embeddings for the senses

#     I created this function to basically help with bruilding data for the TSNE plot
#     """

#     senses = []
#     embeddings = []
    
#     similar_words = [word]
#     for synset in wn.synsets(word):
#         for lem in synset.lemma_names():
#             similar_words.append(lem)


#     for sense in w2v_model.vocab.keys():
#         sense_word = sense.split("_bn:")[0]

#         if sense_word in similar_words:
#             embeddings.append(w2v_model[sense])
#             senses.append(sense)

#         if (len(senses) >= count): break

#     return senses, np.array(embeddings)



# def tsne_similar_words_plot(title: str, labels: List[str], embedding_clusters: np.ndarray, word_clusters: List, alph, filename=None) -> None:
#     """
#     :param title; Chart title
#     :param labels; List of word labels
#     :param embedding_clusters; A numpy array of TSNE fitted embeddings
#     :param word_clusters; A list of words with corressponding embeddings in embedding_clusters
#     :param alph; Alpha value for the scatter plot
#     :param filename; Optional path to save the plot to

#     Reference: https://github.com/sismetanin/word2vec-tsne
#     """
#     plt.figure(figsize=(16, 9))
#     colors = cm.rainbow(np.linspace(0, 1, len(labels)))
#     for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
#         x = embeddings[:, 0]
#         y = embeddings[:, 1]
#         plt.scatter(x, y, c=color, alpha=alph, label=label)
#         for i, word in enumerate(words):
#             plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
#                          textcoords='offset points', ha='right', va='bottom', size=8)
#     plt.legend(loc=4)
#     plt.title(title)
#     plt.grid(True)
#     if filename:
#         plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
#     plt.show()