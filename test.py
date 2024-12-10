import os
# import sys

# from src.processing import mean_pooling, mean_pooling_embedding_with_normalization

# sys.path.append('.')

import argparse
import json
import numpy as np

import faiss
import torch
# from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from abc import abstractmethod

def mean_pooling(tokenEmbeddings, paddingInfo):
    tokenEmbeddingsNoPad = tokenEmbeddings.masked_fill(~paddingInfo[...,None].bool(), 0)
    sentenceEmbeddings = tokenEmbeddingsNoPad.sum(dim = 1) / paddingInfo.sum(dim = 1)[...,None]
    return sentenceEmbeddings

def mean_pooling_embedding_with_normalization(batch_str, tokenizer, model):
    mps_device = torch.device("mps") 
    encoding = tokenizer(batch_str, padding=True, truncation=True, return_tensors='pt').to(mps_device)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    input_ids = input_ids.to(mps_device)
    attention_mask = attention_mask.to(mps_device)
    outputs = model(input_ids, attention_mask=attention_mask)
    sentenceEmbeddings = mean_pooling(outputs[0], attention_mask)
    sentenceEmbeddingsNorm = sentenceEmbeddings.divide(torch.linalg.norm(sentenceEmbeddings,dim = 1)[...,None])
    return sentenceEmbeddingsNorm

class DocumentRetriever:
    @abstractmethod
    def rank_docs(self, query: str, top_k: int):
        """
        Rank the documents in the corpus based on the given query
        :param query:
        :param top_k: 
        :return: ranks and scores of the retrieved documents
        """
class DPRRetriever(DocumentRetriever):
    def __init__(self, model_name: str, faiss_index: str, corpus):
        """

        :param model_name:
        :param faiss_index: The path to the faiss index
        """
        device  = torch.device("mps") 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.faiss_index = faiss_index
        self.corpus = corpus
        self.device = device

    def rank_docs(self, query: str, top_k: int):
        # query_embedding = mean_pooling_embedding(query, self.tokenizer, self.model, self.device)
        with torch.no_grad():
            query_embedding = mean_pooling_embedding_with_normalization(query, self.tokenizer, self.model).detach().cpu().numpy()
        inner_product, corpus_idx = self.faiss_index.search(query_embedding, top_k)
        return corpus_idx.tolist()[0], inner_product.tolist()[0]

#either musique dataset or 2wikimultihopqa dataset
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type = str)
    parser.add_argument('--max_steps',type = str)
    parser.add_argument('--num_demo',type = str)


    args = parser.parse_args()

    dim = 768
    #normalize embeddings before building index using inner product. Note that maximal inner product with normalized embeddings is equivalent to cosine similarity 
    norm = True


    dataset = args.dataset


    model_label = 'facebook_contriever'
    vector_path = f'data/{dataset}/{dataset}_{model_label}_proposition_vectors_norm.npy'
    index_path = f'data/{dataset}/{dataset}_{model_label}_proposition_ip_norm.index'
    if(os.path.isfile(vector_path)):
        vectors = np.load(vector_path)
    if dataset == 'musique':
        corpus = json.load(open('data/musique_proposition_corpus.json', 'r'))
    elif dataset == '2wikimultihopqa':
        corpus = json.load(open('data/2wikimultihopqa_proposition_corpus.json', 'r'))
    corpus_contents = []
    for item in corpus:
        corpus_contents.append(item['title'] + '\n' + item['propositions'])
    print('corpus size: {}'.format(len(corpus_contents)))

    #create sentence-level embeddings using mean-pooling and normalize to prepare for cosine similarity indexing

    if os.path.isfile(vector_path):
        print('Loading existing vectors:', vector_path)
        vectors = np.load(vector_path)
        print('Vectors loaded:', len(vectors))

    else:
        # load model
        tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        model = AutoModel.from_pretrained('facebook/contriever')
        # Check if multiple GPUs are available and if so, use them all
        if (torch.mps.is_available()):
            device = torch.device("mps")    
            model.to(device)
            model = torch.nn.DataParallel(model)
        #test batch size = 16 and batch size = 32 
        batch_size = 16
        vectors = np.zeros((len(corpus_contents), dim))
        #get batch_size number of entries from corpus_contents, tokenize and embed them in 768 dimensional space
        for idx in range(0, len(corpus_contents), batch_size):
            end_idx = min(idx + batch_size, len(corpus_contents))
            seqs = corpus_contents[idx:end_idx]
            try:
                #read above comments to understand what this function does
                batch_embeddings = mean_pooling_embedding_with_normalization(seqs, tokenizer, model)
            except Exception as e:
                batch_embeddings = torch.zeros((len(seqs), dim))
                print(f'Error at {idx}:', e)
            vectors[idx:end_idx] = batch_embeddings.detach().to('cpu').numpy()
        print("Type of vectors is {}".format(type(vectors)))
        fp = open(vector_path, 'wb')
        np.save(fp, vectors)
        fp.close()
        print('vectors saved to {}'.format(vector_path))

        #using FAISS on CPU (GPU support unavailable for mac)
        if os.path.isfile(index_path):
                print('index file already exists:', index_path)
                print('index size: {}'.format(faiss.read_index(index_path).ntotal))
        else:
            print('Building index...')
            index = faiss.IndexFlatIP(dim)
            vectors = vectors.astype('float32')
            index.add(vectors)

            # save faiss index to file
            # fp = open(index_path, 'w')
            faiss.write_index(index, index_path)
            print('index saved to {}'.format(index_path))
            print('index size: {}'.format(index.ntotal))

