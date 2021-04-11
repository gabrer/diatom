#!/usr/bin/env python

# General
import os
import sys
import argparse
import datetime
import time
import math
import random
import numpy as np

# import ipdb
# import gc 

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler
from torch.nn import init
from torch.utils import data
from torch.autograd import Variable

# Collection of optimizer
import torch_optimizer as optim

# Gensim
import gensim

# Creating the dataset
from data_loader import Dataset 
from data_loader import Gensim_Dataset
from data_loader import load_pretrained_embedding
from data_loader import create_dataset_IDMB_partitions_and_labels 
from data_loader import create_dataset_20newsgroup_partitions_and_labels
from data_loader import create_dataset_IMDB_MPST_partitions_and_labels
from data_loader import create_dataset_Amazon_partitions_and_labels
from data_loader import create_dataset_Goodreads
from data_loader import export_sents2json, read_annotated_sentences
from data_loader import compute_SIF_sentence_embedding, compute_AVG_sentence_embedding, compute_BERT_sentence_embedding

# Personal utiles
from utils import write_file
from utils import countElement, print_MultipleSubGraphs_ListOfLosses
from utils import str2bool, Experiment, print_graph, print_graph_single, print_ListOfLosses, print_topicUniqueness

# Tensorboard
import atexit 
import subprocess 
from logger import Logger
from tensorboardX import SummaryWriter

from vae_model import VaeModel
from vae_avitm_paper import VaeAvitmModel
from adversarial_vae_model import AdversarialVaeModel

# Spacy
import spacy
nlp = spacy.load('en')

# Download nltk.stopwords
import nltk
import string
from nltk.corpus import stopwords
nltk.download('stopwords')
stopW = stopwords.words('english') + list(string.punctuation)

# Deadctive logging for Sentence-BERT
import logging
logging.disable(sys.maxsize)



###############################################################################
# --- Configuration parameters ---
###############################################################################
parser = argparse.ArgumentParser()
# Exp options
parser.add_argument('--batchSize',        type=int,   default=64,     required=True)
parser.add_argument('--max_epochs',       type=int,   default=50,     required=True)
parser.add_argument('--sentFrozenEpoch',  type=int,   default=50,     required=True)
parser.add_argument('--lr',               type=float, default=0.001,  required=True)
parser.add_argument('--dr',               type=float, default=0.2,    required=True)
parser.add_argument('--numSentsPerTopic', type=int,   default=0.2,    required=True)
parser.add_argument('--seed',             type=int,   default=1,      metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--opt',              type=str,   default='ADAM',
                    help='Optimizer (e.g. SGD, Adam, etc.)')
parser.add_argument("--gpu",              type=str2bool, nargs='?',
                    const=True,           default=True,  help="Activate GPU.")
parser.add_argument("--tensorboard",      type=str2bool, nargs='?',
                    const=True,           default=False, help="Activate Tensorboard.")

# Paths
parser.add_argument('--dataset_name',          type=str,   required=True)
parser.add_argument('--emb_path',              type=str,   required=True)
parser.add_argument('--ft_emb_path',           type=str,   required=True)
parser.add_argument('--amzPath',               type=str,   required=True)
parser.add_argument('--imdbPath',              type=str,   required=True)
parser.add_argument('--mpstPath',              type=str,   required=True)
parser.add_argument('--goodReadsPlotPath',     type=str,   required=True)
parser.add_argument('--goodReadsReviewsPath',  type=str,   required=True)
parser.add_argument('--annotated_dataset_path',type=str,   required=True)
parser.add_argument('--json_reviews_path',     type=str,   required=True)
parser.add_argument('--json_plots_path',       type=str,   required=True)

# Model options
parser.add_argument('--vae_hid_size',    type=int,   default=50,  required=True)
parser.add_argument('--num_enc_layers',  type=int,   default=2,   required=True)
parser.add_argument('--num_gen_layers',  type=int,   default=0,   required=True)
parser.add_argument('--numOpinionTopics',type=int,   default=50,  required=True)
parser.add_argument('--numPlotTopics',   type=int,   default=50,  required=True)
parser.add_argument('--vocab_size',      type=int,   default=1000,required=True)
parser.add_argument('--de_sparsity',     type=float, default=0.2, required=True)
parser.add_argument('--betaA',           type=float, default=1.0, required=True)
parser.add_argument('--betaS',           type=float, default=1.0, required=True)
parser.add_argument('--embType',         type=str,   default="glove", required=True)
parser.add_argument("--lemmatize",       type=str2bool, nargs='?',
                                         const=False, default=False, help="Lemmitize documents.")
parser.add_argument("--plug_Plots",      type=str2bool, nargs='?',
                                         const=True, default=False, help="Auxiliary VAE with plots.")
parser.add_argument('--z_inter',         type=str, default="concat",
                    help="Interaction between the hidden Z vectors (e.g. concatenate, dot prod., onlySent or onlyNeutral.)")
parser.add_argument('--sent_classi_hid_size', type=int,   default=25,  required=True,
                    help="Hidden size of sentiment classifier.")
parser.add_argument('--plot_classi_hid_size', type=int,   default=25,  required=True,
                    help="Hidden size of plot classifier.")

args = parser.parse_args()
print(args)


def init_optimization(model, optimizer, learning_rate, momentum):
    # Filter out paramteres which does not require optimization (to avoid to update the embedding)
    # advOpt_parameters  = filter(lambda p: p.requires_grad, adv_vae_model.parameters())

    if optimizer == "SGD":
        print("-- SDG optimizer --")
        adv_optimizer   = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum= momentum)
    elif optimizer == "ADAM":
        print("-- ADAM optmizer --")
        adv_optimizer   = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, betas=(0.9, 0.999))
    elif optimizer == "AdaBound":
        adv_optimizer = optim.AdaBound( filter(lambda p: p.requires_grad, model.parameters()),
                                        lr= learning_rate,
                                        betas= (0.9, 0.999),
                                        final_lr = 0.1,
                                        gamma=1e-3,
                                        eps= 1e-8,
                                        weight_decay=0,
                                        amsbound=False,
                                      )
    # Adaptive learning rate
    # scheduler_lr_decay_sent = torch.optim.lr_scheduler.MultiStepLR(sent_optmizer, milestones=[0,1], gamma=0.7)

    return adv_optimizer


###############################################################################
# --- Setting up (hyper)parameters ---
###############################################################################
# Experiment folder
exp           = Experiment(tensorboard=args.tensorboard)
exp.args      = args
momentum      = 0.9
log_interval  = 50

# -- Deterministic run-- 
# os.environ['PYTHONHASHSEED'] = str(args.seed)
# np.random.seed(args.seed)
# random.seed(args.seed)
# torch.backends.cudnn.enabled = False
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)
# if args.gpu:
#     torch.cuda.manual_seed(args.seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# torch.device object used throughout this script
device = torch.device("cuda" if args.gpu else "cpu")  
###############################################


###############################################################################
# Load data
###############################################################################
# --- IMDB_MPST ---
if args.dataset_name   == 'imdb':
    dataset, plot_dataset = create_dataset_IMDB_MPST_partitions_and_labels(exp, size_of_vocab_bow=args.vocab_size)

# --- Amazon Movies ---
elif args.dataset_name == 'amazon':
    dataset, plot_dataset = create_dataset_Amazon_partitions_and_labels(exp, size_of_vocab_bow=args.vocab_size)

# --- Goodreads dataset ---
elif args.dataset_name == 'goodreads':
    dataset, plot_dataset = create_dataset_Goodreads(exp, size_of_vocab_bow=args.vocab_size)

# --- Load word embedding and generate SIF embedding ---
dataset["wordEmb"]      = load_pretrained_embedding(args)

# --- Load annotated sentences dataset ---
annotated_sents_dataset = read_annotated_sentences(dataset, plot_dataset, args.annotated_dataset_path)

# - BERT Sentence Embedding
if args.embType == "bert":
    compute_BERT_sentence_embedding(dataset, annotated_sents_dataset)
elif args.embType == "glove" or args.embType == "fasttext":
    compute_SIF_sentence_embedding(args, dataset, rmpc_svd=[0,1], annotated_sents_dataset=annotated_sents_dataset)

preprocessed            = dataset["preprocessed"] 
documents_bow           = dataset["documents_bow"]    
id2token                = dataset["id2token"]   
token2id                = dataset["token2id"]  
tokenized_corpus        = dataset["tokenized_corpus"]   
docs_of_sents_withIDs   = dataset["docs_of_sents_withIDs"] 
docs_of_sents_withWords = dataset["docs_of_sents_withWords"]
partition               = dataset["partition"]     
labels                  = dataset["labels"]    
id2label                = dataset["id2label"]    
label2id                = dataset["label2id"]    
num_of_classes          = len(label2id)

# Parameters
params = {'batch_size'  : args.batchSize,
          'shuffle'     : True,
          'num_workers' : 2}

# Generators
training_set       = Dataset(documents_bow, partition['train'], labels, args.plug_Plots, plot_dataset["documents_bow"])
training_generator = data.DataLoader(training_set, **params, pin_memory=True)
valid_set          = Dataset(documents_bow, partition['validation'], labels, args.plug_Plots, plot_dataset["documents_bow"])
valid_generator    = data.DataLoader(valid_set, **params, pin_memory=True)
test_set           = Dataset(documents_bow, partition['test'], labels, args.plug_Plots, plot_dataset["documents_bow"])
test_generator     = data.DataLoader(test_set, **params, pin_memory=True)

# Generate dataset objects for Gensim
gensim_dataset     = Gensim_Dataset(documents_bow, partition, tokenized_corpus, id2token)


###############################################################################
# Build the model
###############################################################################
# - Variational Autoencoder Model -
print("Initializing a new VAE Model...")
model   = AdversarialVaeModel(exp, args.vocab_size, num_of_classes=num_of_classes, vae_hidd_size=args.vae_hid_size, 
                                    num_of_OpinionTopics=args.numOpinionTopics, num_of_PlotTopics=args.numPlotTopics, 
                                    encoder_layers=args.num_enc_layers, generator_layers=args.num_gen_layers, beta_s=args.betaS, 
                                    beta_a=args.betaA, encoder_dropout=False, dropout_prob=args.dr, generator_shortcut=False, 
                                    generator_transform=False, interaction=args.z_inter, plug_Plots=args.plug_Plots, device=device).to(device)

# - Optimizer -
adv_optimizer = init_optimization(model, args.opt, args.lr, momentum)


###############################################################################
# Training code
###############################################################################
# @profile
def train(epoch):
    model.train()
    train_loss_epoch   = 0
    losses_compo_epoch = []
    global adv_optimizer

    # ipdb.set_trace()
    for batch_idx, (DocTerm_batch, labels_batch) in enumerate(training_generator):
        DocTerm_batch = DocTerm_batch.float().to(device, non_blocking=True)
        adv_optimizer.zero_grad()

        #-- Model forward pass --
        list_of_computed_params = model(DocTerm_batch)

        # -- Loss --
        loss_components    = model.loss(list_of_computed_params, DocTerm_batch, labels_batch)
        loss               = loss_components[0]
        train_loss_epoch  += loss.item()
        if len(losses_compo_epoch) == 0:
            losses_compo_epoch = loss_components
        else:
            losses_compo_epoch = [previous_l+new_l for previous_l, new_l in zip(losses_compo_epoch, loss_components)]
        loss.backward()

        adv_optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(DocTerm_batch), len(training_generator),
                100. * batch_idx / len(training_generator),
                loss.item()))

    # exp.tensorboard_log(mean_loss_over_batch, model, epoch, "train")
    mean_loss_over_batch = train_loss_epoch / len(training_generator)
    losses_compo_epoch   = [l/len(training_generator) for l in losses_compo_epoch]            

    return (mean_loss_over_batch, losses_compo_epoch)
    

# @profile
def validate(epoch):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (DocTerm_batch, labels_batch) in enumerate(valid_generator):
            DocTerm_batch = DocTerm_batch.float().to(device, non_blocking=True)

            #-- Model forward pass --
            list_of_computed_params = model(DocTerm_batch)

            # -- Loss --
            loss_components = model.loss(list_of_computed_params, DocTerm_batch, labels_batch)
            val_loss       += loss_components[0].item()

    # exp.tensorboard_log(val_loss, model, epoch, "validation")
    val_loss /= len(valid_generator)

    return val_loss


# @profile
def test(epoch):
    model.eval()
    test_loss       = 0
    total_samples   = 0
    overall_correct = 0
    with torch.no_grad():
        for i, (DocTerm_batch, labels_batch) in enumerate(test_generator):
            DocTerm_batch = DocTerm_batch.float().to(device, non_blocking=True)

            #-- Model forward pass --
            list_of_computed_params = model(DocTerm_batch)

            # -- Loss --
            loss_components = model.loss(list_of_computed_params, DocTerm_batch, labels_batch, test=True)
            test_loss      += loss_components[0].item()

            # Compute sentiment classifier accuracy
            y_sent_pred      = torch.max(loss_components[-1], 1)[1]
            total_samples   += labels_batch.size(0)
            overall_correct += (y_sent_pred == labels_batch.to(device)).sum().item()

    # exp.tensorboard_log(test_loss, model, epoch, "test")
    test_loss    /= len(test_generator)
    test_accuracy = overall_correct/total_samples

    return [test_loss, test_accuracy]


# @profile
def main():

    train_losses                   = []
    val_losses                     = []
    test_losses                    = []
    loss_components_list           = []
    test_accuracy_list             = []
    plot_topicTypeCoher_list       = []
    opinion_topicTypeCoher_list    = []
    topic_uniquenessOverall_list   = []
    topic_uniquenessNeutral_list   = []
    topic_uniquenessOpinion_list   = []
    topicCh_mass_avgs              = []
    topicCh_cv_avgs                = []
    topic_purity_list              = []
    max_plot_topic_rate            = -1
    max_topic_purity               = -1
    tCoh_Neutral_over_Epochs       = []
    tCoh_Opinion_over_Epochs       = []

    for epoch in range(1, args.max_epochs + 1):

        exp.epoch = epoch
        t1        = datetime.datetime.utcnow()

        # Training different parts in different moments:
        # https://github.com/alinlab/Confident_classifier/blob/master/src/run_joint_confidence.py
        # global adv_optimizer
        # if epoch >=  args.sentFrozenEpoch:
        #     print("/Frozen\\")
        #     model.alsoAspectLoss = True
        #     model.alsoSentLoss   = False
        #     model.freeze_sent_discriminators(True)
        #     adv_optimizer = init_optimization(model, args.opt, args.lr, momentum)
        # else:
        #     print("/Unfrozen\\")
        #     model.alsoAspectLoss = False
        #     model.alsoSentLoss   = True
        #     model.freeze_sent_discriminators(False)
        #     adv_optimizer = init_optimization(model, args.opt, args.lr, momentum)

        # - Ablation Study -
        # global adv_optimizer
        # print("/// Training  VAE only \\\\\\")
        # model.alsoAspectLoss = False
        # model.alsoSentLoss   = False
        # model.freeze_sent_discriminators(True)
        # model.freeze_plot_vae_and_discriminators(True)
        # model.remove_ortoghonalization_regularizer(True)
        # model.freeze_aspect_sent_VAE_encoders(False)
        # adv_optimizer = init_optimization(model, args.opt, args.lr, momentum)

        batchMeanLoss, loss_components  =  train(epoch)
        v_loss                          =  validate(epoch)
        test_loss, test_accuracy        =  test(epoch)

        print("====> Epoch train completed in ", datetime.datetime.utcnow()-t1)

        # Loss
        train_losses.append(float(batchMeanLoss))
        val_losses.append(float(v_loss))
        test_losses.append(test_loss)
        loss_components_list.append(loss_components[1:])
        print_graph(train_losses, val_losses, "Train-Val loss", 'Train', 'Validation', exp.exp_folder_path+"loss_Validation")
        print_graph(train_losses, test_losses, "Train-Test loss", 'Train', 'Test', exp.exp_folder_path+"loss_Testset")
        print_ListOfLosses(loss_components_list, val_losses, "List of Train-Val loss", exp.exp_folder_path+"list_of_loss")
        print_MultipleSubGraphs_ListOfLosses(loss_components_list, exp.exp_folder_path+"list_of_losses_SubGraph")

        # Testset accuracy
        test_accuracy_list.append(test_accuracy)
        print_graph(val_losses, test_accuracy_list, "Accuracy on testset - Loss on Validation", 'Val Loss', 'Test Accuracy', 
                    exp.exp_folder_path+"accuracy_Testset", miny=0.0, maxy=1.0)

        # Topic-Type Coherence
        actual_opinion_topic_rate, actual_plot_topic_rate = model.compute_topic_type_coherence_statistics(dataset, annotated_sents_dataset)  
        plot_topicTypeCoher_list.append(actual_plot_topic_rate)
        opinion_topicTypeCoher_list.append(actual_opinion_topic_rate)
        print_graph(plot_topicTypeCoher_list, opinion_topicTypeCoher_list, "Disentangling rate", 'Plot topics', 'Opinion Topics', 
                    exp.exp_folder_path+"topic_disentaRate", miny=0.0, maxy=1.0)

        # TopicType coherence with new peaks
        new_topicTypeCoh_peak = False
        if actual_plot_topic_rate > max_plot_topic_rate:
            max_plot_topic_rate  = actual_plot_topic_rate
            new_topicTypeCoh_peak = True

        # Topic Purity  
        purity = model.compute_topic_purity_statistics(dataset, annotated_sents_dataset)
        topic_purity_list.append(purity)
        new_topicPurity_peak = False
        if purity >= max_topic_purity:
            max_topic_purity = purity
            new_topicPurity_peak = True

        # Topic Uniqueness
        topic_uniquenessOverall_list.append(model.t_uniqueness(dataset, annotated_sents_dataset, topicType='All'))
        topic_uniquenessNeutral_list.append(model.t_uniqueness(dataset, annotated_sents_dataset, topicType='Neutral'))
        topic_uniquenessOpinion_list.append(model.t_uniqueness(dataset, annotated_sents_dataset, topicType='Opinion'))
        print_topicUniqueness(args, topic_uniquenessOverall_list, topic_uniquenessNeutral_list, topic_uniquenessOpinion_list, 
                               "Topic Uniqueness", exp.exp_folder_path+"Topics_Uniqueness")

        if epoch % 10 == 0 or epoch == 1 or new_topicTypeCoh_peak or new_topicPurity_peak:
            with torch.no_grad():

                # Print Topic Coherence
                avg_tCh_mass_list, avg_tCh_cv_list = model.print_topics(dataset, gensim_dataset, id2token, visualized_words=10, annotated_sents_dataset=annotated_sents_dataset)
                topicCh_mass_avgs.append(avg_tCh_mass_list)
                topicCh_cv_avgs.append(avg_tCh_cv_list)
                print_graph_single(epoch, topicCh_mass_avgs, "Topic Coherences", exp.exp_folder_path+"topicCh_mass", autoscale=True)
                print_graph_single(epoch, topicCh_cv_avgs,   "Topic Coherences", exp.exp_folder_path+"topicCh_cv")

                # Print topic labels (i.e. topic sentences)
                model.print_TopicSentences(exp, dataset, epoch, export=False, annotated_sents_dataset=annotated_sents_dataset)

                # --- Statistics about highest values so far ---
                # Topic coherence max
                tCoh_Neutral_over_Epochs.append(avg_tCh_cv_list[0])
                tCoh_Opinion_over_Epochs.append(avg_tCh_cv_list[1])

                # --- Visualization via T-SNE ---
                model.visualize_topics_tsne(exp, dataset, annotated_sents_dataset=annotated_sents_dataset)

        # # Print final experiment statistics
        # if epoch == args.max_epochs:
        #     id_neutralTCh  = np.argmax(np.array(tCoh_Neutral_over_Epochs))
        #     id_opinionTCh  = np.argmax(np.array(tCoh_Opinion_over_Epochs))
        #     id_topicUniqu  = np.argmax(np.array(topic_uniquenessOverall_list))
        #     id_topicPurity = np.argmax(np.array(topic_purity_list))
        #     id_TestAcc     = np.argmax(np.array(test_accuracy_list))
        #     id_PerxTest    = np.argmin(np.array(perplexity_testset_list))
        #     print("\n\n==> Max Overall Topic Coherence: ", (tCoh_Neutral_over_Epochs[id_neutralTCh]+tCoh_Opinion_over_Epochs[id_opinionTCh]) / 2)
        #     print("Max Neutral Topic Coherence: ", tCoh_Neutral_over_Epochs[id_neutralTCh], id_neutralTCh)
        #     print("Max Opinion Topic Coherence: ", tCoh_Opinion_over_Epochs[id_opinionTCh], id_opinionTCh)
        #     print("==> MAX Uniqueness all topics:", topic_uniquenessOverall_list[id_topicUniqu], id_topicUniqu)
        #     print("==> Max Purity: ", topic_purity_list[id_topicPurity], id_topicPurity)
        #     print("==> Max Test accuracy: ", test_accuracy_list[id_TestAcc], id_TestAcc)    
        #     print("==> Min Perplexity on Testset: ", perplexity_testset_list[id_PerxTest], id_PerxTest)    


if __name__ == "__main__":
    main()

        
