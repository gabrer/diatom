#!/usr/bin/env python

import sys
from utils import countElement
from collections import Counter
from collections import defaultdict


class Topic:

    def __init__(self, list_of_words, topic_type=None, list_of_weights=None, 
                       list_of_sentences=None, list_sentence_scores=None, 
                       list_of_sentence_labels=None, list_of_plotIDs=None,
                       label2text=None, whole_WordDistr=None):

        self.words_in_topic          = list_of_words
        self.whole_WordDistr         = whole_WordDistr  #
        self.word_weights            = list_of_weights 
        self.list_of_sentences       = list_of_sentences
        self.list_sentence_scores    = list_sentence_scores
        self.list_of_plotIDs         = list_of_plotIDs
        self.list_of_sentence_labels = list_of_sentence_labels
        self.topic_type              = topic_type

        self.label2text = label2text

        self.most_common_label       = None
        self.freq_of_most_comm_label = None
        self.topic_type_coherence    = None


    # TopicType Coherence is similar to cluster purity but just for the single cluster/topic.
    def compute_most_common_label(self):

        if self.label2text is None:
            print("ERR: Sentence labels for the current topic are not defined.")
            sys.exit(0)

        if not hasattr(self, 'text2label'):
            l2t = self.label2text
            self.text2label = {l2t[index]:index  for index in range(len(l2t))}

        self.list_of_sentence_labels = [sent_svd.label for sent_svd in self.list_of_sentences]

        # Count label occurences
        labelCounter = countElement(l for sent_labels in self.list_of_sentence_labels for l in sent_labels)
        # Set default value to 0 for keys not in counter
        labelCounter = defaultdict(lambda:0, labelCounter)

        num_of_opinion_labels = labelCounter[self.text2label["Positive"]]+labelCounter[self.text2label["Negative"]]
        num_of_neutral_labels = labelCounter[self.text2label["Plot"]]+labelCounter[self.text2label["None"]]

        if  num_of_opinion_labels > num_of_neutral_labels:
            self.most_common_label       = self.text2label["Positive"]
            self.freq_of_most_comm_label = num_of_opinion_labels
        else:
            self.most_common_label       = self.text2label["Plot"]
            self.freq_of_most_comm_label = num_of_neutral_labels


    def compute_topic_type_coherence(self):
        # Compute and the needed attribute of the class to compute the topictype coherence
        self.compute_most_common_label()
        self.topic_type_coherence    = self.freq_of_most_comm_label / len(self.list_of_sentence_labels)


    def print_topic_details(self, log_file=None):
        topic_details = ""
        topic_details += "\n"+"----- "+str(self.topic_type)+" -----"+"\n"

        # Print topic words
        topic_details += "  ".join(self.words_in_topic)+"\n"

        # Print word weights
        if self.word_weights is not None:
            topic_details += " ".join([f"{w:.3f}" for w in self.word_weights])
        topic_details += "\n"
       
        if self.label2text is not None:
            # Compute topic statistics
            if self.most_common_label is None:
                self.compute_topic_type_coherence()

            # Print sentence labels
            for sent_svd in self.list_of_sentences:
                label_string = " ".join( [self.label2text[l] for l in sent_svd.label] )
                topic_details += ("{:01.3f} [{:<8}| {:<4}] {:>}\n").format(sent_svd.score, label_string, str(sent_svd.plotID), str("".join(sent_svd))) 

            # Print most common labels
            if self.label2text[self.most_common_label] == "Positive":
                str_type = "Opinion"
            else:
                str_type = "Plot/None"
            topic_details += str_type + "  " + str(self.topic_type_coherence) +"\n"
        
        else:
            # Print sentence labels for dataset without annotation
            for sent_svd in self.list_of_sentences: 
                topic_details +=  "\u2022 "+" ".join(sent_svd)+"\n"

        return topic_details
