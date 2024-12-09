import os
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
    if(torch.mps.is_available() and torch.mps.device_count() > 0):
        device = torch.device("mps") 
    else:
        device = torch.device("cpu")
    # add to mps device
    encoding = tokenizer(batch_str, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    #add to mps device on these next two
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
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
        if(torch.cuda.is_available() and torch.cuda.device_count > 0):
            # uses 1 GPU
            print("Running on 1 GPU")
            device = torch.device("mps")
            # model = model.to(device)
            # model = torch.nn.DataParallel(model)
        else:
            print("Running on CPU")
            device = torch.device("cpu")
            # model = model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.faiss_index = faiss_index
        self.corpus = corpus
        self.device = device

    def rank_docs(self, query: str, top_k: int):
        # query_embedding = mean_pooling_embedding(query, self.tokenizer, self.model, self.device)
        with torch.no_grad():
            #must go to cpu after detach
            query_embedding = mean_pooling_embedding_with_normalization(query, self.tokenizer, self.model).detach().cpu().numpy()
        inner_product, corpus_idx = self.faiss_index.search(query_embedding, top_k)
        return corpus_idx.tolist()[0], inner_product.tolist()[0]
    
def retrieve_step(query: str, corpus, top_k: int, retriever: DocumentRetriever, dataset: str):
    doc_ids, scores = retriever.rank_docs(query, top_k=top_k)
    if dataset in ['hotpotqa']:
        retrieved_passages = []
        for doc_id in doc_ids:
            key = list(corpus.keys())[doc_id]
            retrieved_passages.append(key + '\n' + ''.join(corpus[key]))
    elif dataset in ['musique', '2wikimultihopqa']:        
        retrieved_passages = [corpus[doc_id]['title'] + '\n' + corpus[doc_id]['text'] for doc_id in doc_ids]
    else:
        raise NotImplementedError(f'Dataset {dataset} not implemented')
    return retrieved_passages, scores

def process_sample(idx, sample, dataset, top_k, k_list,max_steps, corpus, retriever, processed_ids):
    # Check if the sample has already been processed
    if dataset in ['hotpotqa', '2wikimultihopqa']:
        sample_id = sample['_id']
    elif dataset in ['musique']:
        sample_id = sample['id']
    else:
        raise NotImplementedError(f'Dataset {dataset} not implemented')
    if sample_id in processed_ids:
        return  # Skip already processed samples
    else:
        processed_ids.add(sample_id)

    # Perform retrieval and reasoning steps
    query = sample['question']
    #uncomment this line if you want to see the questions being asked
    # print(query)
    retrieved_passages, scores = retrieve_step(query, corpus, top_k, retriever, dataset)


    # calculate recall
    if dataset in ['musique']:
        gold_passages = [item for item in sample['paragraphs'] if item['is_supporting']]
        # print(gold_passages)
        gold_items = set([item['title'] + '\n' + item['paragraph_text'] for item in gold_passages])
        # print(gold_items)
        retrieved_items = retrieved_passages
        # print(retrieved_passages:10)
    elif dataset in ['2wikimultihopqa']:
        gold_passages = [item for item in sample['supporting_facts']]
        gold_items = set([item[0] for item in gold_passages])
        retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
    else:
        raise NotImplementedError(f'Dataset {dataset} not implemented')

    recall = dict()
    print(f'idx: {idx + 1} ', end='')
    for k in k_list:
        # in the top k retrieved docs, sum the number of true positives (gold items) found and divide by the total number of true positives
        # fraction of retrieved passages found
        recall[k] = sum(1 for t in gold_items if t in retrieved_items[:k]) / len(gold_items)
    return idx, recall, retrieved_passages

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

    #change to /scratch/gpfs/<YourNetID>
    # vector_path = f'data/{dataset}/{dataset}_{model_label}_proposition_vectors_norm.npy'
    # index_path = f'data/{dataset}/{dataset}_{model_label}_proposition_ip_norm.index'

    if dataset == 'musique':
        corpus = json.load(open('data/musique_proposition_corpus.json', 'r'))
    elif dataset == '2wikimultihopqa':
        corpus = json.load(open('data/2wikimultihopqa_proposition_corpus.json', 'r'))
    corpus_contents = []
    for item in corpus:
        corpus_contents.append(item['title'] + '\n' + item['propositions'])


    #load in file available?
    # if os.path.isfile(vector_path):
    #     print('Loading existing vectors:', vector_path)
    #     vectors = np.load(vector_path)
    #     print('Vectors loaded:', len(vectors))


    # else:
        # load model
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    model = AutoModel.from_pretrained('facebook/contriever')
    # Check if multiple GPUs are available and if so, use them all
    if(torch.mps.is_available() and torch.mps.device_count() > 0):
        # uses 1 GPU
        print("Running on 1 GPU")
        device = torch.device("mps")
        model = model.to(device)
        # model = torch.nn.DataParallel(model)
    else:
        print("Running on CPU")
        device = torch.device("cpu")
        model = model.to(device)
    #test batch size = 16 and batch size = 32 (should be > # GPU's used)
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
        #change to go to the cpu
        vectors[idx:end_idx] = batch_embeddings.detach().cpu().numpy()

    print("Type of vectors is {}".format(type(vectors)))
    # fp = open(vector_path, 'wb')
    # np.save(fp, vectors)
    # fp.close()
    # print('vectors saved to {}'.format(vector_path))

    #using FAISS on CPU (GPU support unavailable for mac)
    # if os.path.isfile(index_path):
    #         print('index file already exists:', index_path)
    #         print('index size: {}'.format(faiss.read_index(index_path).ntotal))
    # else:
    print('Building index...')
    index = faiss.IndexFlatIP(dim)
    vectors = vectors.astype('float32')
    index.add(vectors)



    llm_model = 'gpt-3.5-turbo-1106'
    llm = 'openai'

    """
    User-inputted args
    """

    """
    For 2wikimultihopqa, change max_steps to 2, num_demo to 1 to perform multistep retrieval.
    For musique, change max_steps to 4, num_demo 1 to perform multistep retrieval.
    """
    max_steps = args.max_steps
    num_demo = args.num_demo
    top_k = 8
    #load dataset
    if dataset == '2wikimultihopqa':
        data = json.load(open('data/2wikimultihopqa.json', 'r'))
        corpus = json.load(open('data/2wikimultihopqa_corpus.json', 'r'))
        prompt_path = 'data/ircot_prompts/2wikimultihopqa/gold_with_3_distractors_context_cot_qa_codex.txt'
        max_steps = max_steps if max_steps is not None else 2
    else:
        raise NotImplementedError(f'Dataset {dataset} not implemented')


    # from langchain_openai import ChatOpenAI
    # from dotenv import load_dotenv
    # load_dotenv()
    #Create OpenAI Client
    # client = ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), model=llm_model, temperature=0.0, max_retries=5, timeout=60)



    # doc_ensemble_str = 'doc_ensemble' if doc_ensemble else 'no_ensemble'

    doc_ensemble_str = ''
    alt_model_label = 'facebook_contriever'

    #change output path to /scratch/gpfs/<YourNetID> 
    if max_steps > 1:
        output_path = f'output/ircot/{dataset}_{alt_model_label}_demo_{num_demo}_{llm_model}_{doc_ensemble_str}_step_{max_steps}_top_{top_k}.json'
    else:  # only one step
        top_k = 100
        # output_path = f'output/proposition_{dataset}_{alt_model_label}_{doc_ensemble_str}.json'

    model_label = 'facebook/contriever'
    retriever = DPRRetriever(model_label, index, corpus)

    k_list = [1, 2, 5, 10, 15, 20, 30, 50, 100]
    total_recall = {k: 0 for k in k_list}

    results = data

    processed_ids = set()

    for idx, sample in enumerate(data):
        idx, recall, retrieved_passages = process_sample(idx, sample, dataset, top_k, k_list,max_steps, corpus, retriever, processed_ids) 


        # print metrics
        for k in k_list:
            total_recall[k] += recall[k]
            print(f'R@{k}: {total_recall[k] / (idx + 1):.4f} ', end='')
        print()

        # record results
        results[idx]['retrieved'] = retrieved_passages
        results[idx]['recall'] = recall

    for k in k_list:
        #average recall (across 1,000 questions for musique)
        print(f'R@{k}: {total_recall[k] / len(data):.4f} ', end='')