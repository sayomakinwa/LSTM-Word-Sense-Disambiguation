from corpora import *
from predict import *
from score import *

# corpora_xml_path = '../data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml'
# gold_mapping_path = '../data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt'
# resources_path = '../resources/'
# outfolder_path = '../data/WSD_Evaluation_Framework/Training_Corpora/SemCor'


# corpora_xml_path = '../data/WSD_Evaluation_Framework/Evaluation_Datasets/senseval3/senseval3.data.xml'
# gold_mapping_path = '../data/WSD_Evaluation_Framework/Evaluation_Datasets/senseval3/senseval3.gold.key.txt'
# resources_path = '../resources/'
# outfolder_path = '../data/WSD_Evaluation_Framework/Evaluation_Datasets/senseval3'

# corpora_xml_path = '../data/WSD_Evaluation_Framework/Training_Corpora/SemCor+OMSTI/semcor+omsti.data.xml'
# gold_mapping_path = '../data/WSD_Evaluation_Framework/Training_Corpora/SemCor+OMSTI/semcor+omsti.gold.key.txt'
# resources_path = '../resources/'
# outfolder_path = '../data/WSD_Evaluation_Framework/Training_Corpora/SemCor+OMSTI'

# extract_training_data(corpora_xml_path, gold_mapping_path, resources_path, outfolder_path)


input_path = '../data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml'
output_path = '../data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007'
resources_path = '../resources'

predict_babelnet(input_path, output_path, resources_path)



prediction = '../data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/pred.txt'
gold_file =  '../data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt'
bn2wn_mapping_file = '../resources/babelnet2wordnet.tsv'
score_predict1(prediction, gold_file, bn2wn_mapping_file)







# input_path = '../data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2013/semeval2013.data.xml'
# output_path = '../data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2013'
# resources_path = '../resources'

# predict_babelnet(input_path, output_path, resources_path)



# prediction = '../data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2013/pred.txt'
# gold_file =  '../data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2013/semeval2013.gold.key.txt'
# bn2wn_mapping_file = '../resources/babelnet2wordnet.tsv'
# score_predict1(prediction, gold_file, bn2wn_mapping_file)




# input_path = '../data/WSD_Evaluation_Framework/Evaluation_Datasets/senseval2/senseval2.data.xml'
# output_path = '../data/WSD_Evaluation_Framework/Evaluation_Datasets/senseval2'
# resources_path = '../resources'

# predict_babelnet(input_path, output_path, resources_path)



# prediction = '../data/WSD_Evaluation_Framework/Evaluation_Datasets/senseval2/pred_babelnet.txt'
# gold_file =  '../data/WSD_Evaluation_Framework/Evaluation_Datasets/senseval2/senseval2.gold.key.txt'
# bn2wn_mapping_file = '../resources/babelnet2wordnet.tsv'
# score_predict1(prediction, gold_file, bn2wn_mapping_file)







# with open("../data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/trainX.txt", "r") as sentences_x:
#   with open("../data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/trainy.txt", "r") as sentences_y:
#       for a, b in zip(sentences_x, sentences_y):
#           if len(a.split()) != len(b.split()):
#               print ("Error!")
#               print ("a: "+a)
#               print ("b: "+b)
#               print ("\n")




# from nltk.corpus import wordnet as wn

# word = "change_ringing"
# synsets = wn.synsets(word)
# sense2freq = {}
# for s in synsets:
#   freq = 0  
#   for lemma in s.lemmas():
#       freq += lemma.count()
#   wn_synset_id = "wn:" + str(s.offset()).zfill(8) + s.pos()
#   sense2freq[wn_synset_id] = freq

# for s in sense2freq:
#   print (s, sense2freq[s])


# mapping = {}
# with open("../resources/babelnet2wndomains.tsv", 'r') as handle:
#     for line in handle:
#             line = line.strip().split("\t")
#             if (line):
#                 if line[1] in mapping:
#                     mapping[line[1]] += 1
#                 else:
#                     mapping[line[1]] = 1

#     most_freq = ""
#     highest_freq = 0

#     for key, value in mapping.items():
#         #print(key, value)
#         if int(value) > highest_freq:
#             most_freq = key
#             highest_freq = value
#             #print(most_freq, highest_freq)

#     print (most_freq)





# Do word embeddings vector arithmetcs with the tnse plot; king + woman = queen ===== 
# model.wv.most_similar(positive=['woman', 'king'], negative=['man'])
# [('queen', 0.50882536), ...]



# [18:06, 4/24/2019] Federico Vergallo: Clone the repo on your pc:
# git clone https://gitlab.com/<username>/<first_name>_<surname>_<matricola>_nlp19hw1.git
# [18:06, 4/24/2019] Federico Vergallo: Then copy all what you need in the folder just created
# [18:07, 4/24/2019] Federico Vergallo: Then go in the folder and do:
# git add --all
# git commit -m "your message"
# git push
# [18:07, 4/24/2019] Federico Vergallo: And you done