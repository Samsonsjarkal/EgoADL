import json
import re
import csv
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet

stop_words_lite = {'', 'a', 'an', 'some', 'the', 'their'}
clean_file = 'check.txt'
def readtxt(filename):
    word_dict = dict()
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            for i in range(int(len(row) / 2)):
                word_dict[row[i*2 + 1]] = row[0]
    return word_dict
            
word_dict = readtxt(clean_file)

def load_json(filename):
    with open(filename) as f:
        data = f.read()

    input_js = json.loads(data)
    return input_js

## pos the tag of each word
def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None

## apply the lemmar for each action
def lemmatize(act):
    global word_dict
    lemmatizer = WordNetLemmatizer()
 
    # Define function to lemmatize each word with its POS tag
    
    # POS_TAGGER_FUNCTION : TYPE 1
    
    # tokenize the sentence and find the POS tag for each token
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(act)) 
    
    # print(pos_tagged)
        
    #>[('the', 'DT'), ('cat', 'NN'), ('is', 'VBZ'), ('sitting', 'VBG'), ('with', 'IN'),
    # ('the', 'DT'), ('bats', 'NNS'), ('on', 'IN'), ('the', 'DT'), ('striped', 'JJ'),
    # ('mat', 'NN'), ('under', 'IN'), ('many', 'JJ'), ('flying', 'VBG'), ('geese', 'JJ')]
    
    # As you may have noticed, the above pos tags are a little confusing.
    
    # we use our own pos_tagger function to make things simpler to understand.
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    # print(wordnet_tagged)
    for word_type in wordnet_tagged:
        if (word_type[0] not in word_dict):
            word_dict[word_type[0]] = word_type[1]
    #>[('the', None), ('cat', 'n'), ('is', 'v'), ('sitting', 'v'), ('with', None),
    # ('the', None), ('bats', 'n'), ('on', None), ('the', None), ('striped', 'a'),
    # ('mat', 'n'), ('under', None), ('many', 'a'), ('flying', 'v'), ('geese', 'a')]
    
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:       
            # else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    lemmatized_sentence = " ".join(lemmatized_sentence)
    
    return lemmatized_sentence

def remove_stop_word(act):
    # print("test:", act)
    act = re.sub(r'/', ' ', act)
    act = re.sub(r'[^A-Za-z0-9 ]+', '', act.lower())
    act_word_list = act.split(" ")
    for word in stop_words_lite:
        while (word in act_word_list):
            act_word_list.remove(word)
    act = ' '.join(act_word_list)
    act = lemmatize(act)
    act = word_dict[act]
    # print(act)
    return act
    
def clean_label(label):
    label_clean = label
    for data in label['data']:
        index = 0
        for act_label in label['data'][data]:
            # print(act_label['label'])
            act_new = remove_stop_word(act_label['label'])
            # print(act_new)
            label_clean['data'][data][index]['label'] = act_new
            index += 1

    return label_clean
    
def main(labelfile):
    label = load_json(labelfile)
    label_clean = clean_label(label)
    labelfile_clean = labelfile.rsplit("\\", 1)[0]
    labelfile_clean += '\\label_clean.AUCVL'
    # with open(labelfile_clean, 'w') as f:
    #     json.dump(label_clean, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', required=True, type=str, help="File path of dataset")
    args = parser.parse_args()
    filepath = args.path

    labelfile = filepath + 'label_sync.AUCVL'
    main(labelfile)
        
    # print(word_dict)
    
    