import asyncio
import os
import random
# import sys

# from src.processing import mean_pooling, mean_pooling_embedding_with_normalization

# sys.path.append('.')

import argparse
import json
import numpy as np
import time
import backoff


import faiss
import torch
# from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from abc import abstractmethod
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
import openai
import tiktoken
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

"""
methods for generating embeddings
"""

def mean_pooling(tokenEmbeddings, paddingInfo):
    tokenEmbeddingsNoPad = tokenEmbeddings.masked_fill(~paddingInfo[...,None].bool(), 0)
    sentenceEmbeddings = tokenEmbeddingsNoPad.sum(dim = 1) / paddingInfo.sum(dim = 1)[...,None]
    return sentenceEmbeddings

def mean_pooling_embedding_with_normalization(batch_str, tokenizer, model):
    encoding = tokenizer(batch_str, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    if(torch.cuda.is_available()):
        cuda_device = torch.device("cuda") 
        input_ids = input_ids.to(cuda_device)
        attention_mask = attention_mask.to(cuda_device)
    else:
        cuda_device = torch.device("cpu") 
        input_ids = input_ids.to(cuda_device)
        attention_mask = attention_mask.to(cuda_device)
    outputs = model(input_ids, attention_mask=attention_mask)
    sentenceEmbeddings = mean_pooling(outputs[0], attention_mask)
    sentenceEmbeddingsNorm = sentenceEmbeddings.divide(torch.linalg.norm(sentenceEmbeddings,dim = 1)[...,None])
    return sentenceEmbeddingsNorm    

"""
end of methods for generating embeddings
"""

"""
Methods for multistep retrieval
"""

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

def parse_prompt(file_path: str, has_context=True):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split the content by the metadata pattern
    parts = content.split('# METADATA: ')
    parsed_data = []
    if has_context:
        for part in parts[1:]:  # Skip the first split as it will be empty
            metadata_section, rest_of_data = part.split('\n', 1)
            metadata = json.loads(metadata_section)
            document_sections = rest_of_data.strip().split('\n\nQ: ')
            document_text = document_sections[0].strip()
            qa_pair = document_sections[1].split('\nA: ')
            question = qa_pair[0].strip()
            thought_and_answer = qa_pair[1].strip().split('So the answer is: ')
            thought = thought_and_answer[0].strip()
            answer = thought_and_answer[1].strip()

            parsed_data.append({
                'metadata': metadata,
                'document': document_text,
                'question': question,
                'thought_and_answer': qa_pair[1].strip(),
                'thought': thought,
                'answer': answer
            })
    else:
        for part in parts[1:]:
            metadata_section, rest_of_data = part.split('\n', 1)
            metadata = json.loads(metadata_section)
            s = rest_of_data.split('\n')
            question = s[0][3:].strip()
            thought_and_answer = s[1][3:].strip().split('So the answer is: ')
            thought = thought_and_answer[0].strip()
            answer = thought_and_answer[1].strip()

            parsed_data.append({
                'metadata': metadata,
                'question': question,
                'thought_and_answer': s[1][3:].strip(),
                'thought': thought,
                'answer': answer
            })

    return parsed_data


def num_tokens_by_tiktoken(text: str):
    return len(enc.encode(text))


# ircot_reason_instruction = 'You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. Your task is to generate the next thought for the current step, DON\'T generate all thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".'
ircot_reason_instruction = 'You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through a demonstration consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts, delineated by the string "Thought". Your task is to generate one thought for current step, DON\'T generate all thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".'

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def make_api_call_to_gpt(prompt, prompt_demo, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "developer", "content": ircot_reason_instruction + '\n\n' + prompt_demo},
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature = 0,
        timeout=60, 
        max_tokens = 400
    )
    return response.choices[0].message.content

def reason_step(dataset, few_shot: list, query: str, passages: list, thoughts: list, client):
    """
    Given few-shot samples, query, previous retrieved passages, and previous thoughts, generate the next thought with LangChain LLM.
    The generated thought is used for further retrieval step.
    :return: next thought
    """
    # #handle rate limit errors
    # seconds_to_pause_after_rate_limit_error = 15
    # seconds_to_sleep_each_loop = (
    #         0.001  # 1 ms limits max throughput to 1,000 requests per second
    # )
    # available_request_capacity = 500
    # available_token_capacity = 200000
    # last_update_time = time.time()

    # queue_of_requests_to_retry = asyncio.Queue()

    # example used for few-shot prompting
    prompt_demo = ''

    #contents from the top k retrieved documents
    prompt_user = ''

        
    #put into the format of documents in IRCOT tests
    for passage in passages:
        prompt_user += f'Wikipedia Title: {passage}\n\n'
    prompt_user += f'Question: {query}\nThought:' + ' '.join(thoughts)

    for sample in few_shot:
        cur_sample = f'{sample["document"]}\n\nQuestion: {sample["question"]}\nThought: {sample["thought_and_answer"]}\n\n'
        # return num_tokens_by_tiktoken(ircot_reason_instruction + prompt_demo + cur_sample + prompt_user)
    #     #token limit
        if num_tokens_by_tiktoken(ircot_reason_instruction + prompt_demo + cur_sample + prompt_user) < 4096:
                # examples.append({"input": f'{sample["document"]}\n\nQuestion: {sample["question"]}\nThought: '+sample["thought"], "output": sample["thought_and_answer"]})
                prompt_demo += cur_sample

    # messages = ChatPromptTemplate.from_messages([("system", ircot_reason_instruction + '\n\n' + prompt_demo),
    #                                              ("human", prompt_user)])

    return make_api_call_to_gpt(prompt_user, prompt_demo)



def process_sample(idx, sample, dataset, top_k, k_list,max_steps, few_shot_samples, corpus, retriever, client, processed_ids):


    
    # Check if the sample has already been processed
    if dataset in ['hotpotqa', '2wikimultihopqa']:
        sample_id = sample['_id']
    elif dataset == 'musique':
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

    thoughts = []
    retrieved_passages_dict = {passage: score for passage, score in zip(retrieved_passages, scores)}
    it = 1
    for it in range(1, max_steps):
        print("in IRCOT loop")
        new_thought = reason_step(dataset, few_shot_samples, query, retrieved_passages[:top_k], thoughts, client)
        # return new_thought
        # print(new_thought)
        thoughts.append(new_thought)
        # print(new_thought)
        if 'So the answer is:' in new_thought:
            break
        new_retrieved_passages, new_scores = retrieve_step(new_thought, corpus, top_k, retriever, dataset)

        for passage, score in zip(new_retrieved_passages, new_scores):
            if passage in retrieved_passages_dict:
                retrieved_passages_dict[passage] = max(retrieved_passages_dict[passage], score)
            else:
                retrieved_passages_dict[passage] = score

        retrieved_passages, scores = zip(*retrieved_passages_dict.items())

        sorted_passages_scores = sorted(zip(retrieved_passages, scores), key=lambda x: x[1], reverse=True)
        retrieved_passages, scores = zip(*sorted_passages_scores)
    # end iteration

    # calculate recall
    if dataset in ['hotpotqa']:
        gold_passages = [item for item in sample['supporting_facts']]
        gold_items = set([item[0] for item in gold_passages])
        retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
    elif dataset in ['musique']:
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
    return idx, recall, retrieved_passages, thoughts, it

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
        if(torch.cuda.is_available()):
            device  = torch.device("cuda") 
        else:
            device  = torch.device("cpu") 
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

            
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type = str)
    parser.add_argument('--max_steps',type = int)
    parser.add_argument('--num_demo',type = int)


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
        """CHANGE THIS FOR SERVER RUN"""
        if (torch.cuda.is_available()):
            device = torch.device("cuda")    
            model.to(device)
        else:
            device = torch.device("cpu")
            model.to(device)
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

        # using FAISS on CPU (GPU support unavailable for mac)
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
            # faiss.write_index(index, index_path)
            # print('index saved to {}'.format(index_path))
            # print('index size: {}'.format(index.ntotal))

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
    if dataset == 'musique':
        data = json.load(open('data/musique.json', 'r'))
        corpus = json.load(open('data/musique_corpus.json', 'r'))
        prompt_path = 'data/ircot_prompts/musique/gold_with_3_distractors_context_cot_qa_codex.txt'
        max_steps = max_steps if max_steps is not None else 4
    elif dataset == '2wikimultihopqa':
        data = json.load(open('data/2wikimultihopqa.json', 'r'))
        corpus = json.load(open('data/2wikimultihopqa_corpus.json', 'r'))
        prompt_path = 'data/ircot_prompts/2wikimultihopqa/gold_with_3_distractors_context_cot_qa_codex.txt'
        max_steps = max_steps if max_steps is not None else 2
    else:
        raise NotImplementedError(f'Dataset {dataset} not implemented')
    
    # doc_ensemble = ''
    # doc_ensemble_str = 'doc_ensemble' if doc_ensemble else 'no_ensemble'
    doc_ensemble_str = ''
    alt_model_label = 'facebook_contriever'
    if max_steps > 1:
        k_list = [1, 2, 5, 8]
        output_path = f'output/ircot/{dataset}_{alt_model_label}_demo_{num_demo}_{llm_model}_{doc_ensemble_str}_step_{max_steps}_top_{top_k}.json'
    else:  # only one step
        top_k = 100
        output_path = f'output/proposition_{dataset}_{alt_model_label}_{doc_ensemble_str}.json'
        k_list = [1, 2, 5, 10, 15, 20, 30, 50, 100]
    
    if dataset == 'musique':
        faiss_index = faiss.read_index('data/musique/musique_facebook_contriever_proposition_ip_norm.index')
    else:
        faiss_index = faiss.read_index('data/2wikimultihopqa/2wikimultihopqa_facebook_contriever_proposition_ip_norm.index')
        
    model_label = 'facebook/contriever'
    retriever = DPRRetriever(model_label, faiss_index, corpus)


    total_recall = {k: 0 for k in k_list}

    results = data
    processed_ids = set()
    
    load_dotenv('.env')
    # print(os.getenv("OPENAI_API_KEY"))
    #Create OpenAI Client
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    few_shot_samples = parse_prompt(prompt_path)
    few_shot_samples = few_shot_samples[:num_demo]
    print('num of demo:', len(few_shot_samples))

    k_list = [1, 2, 5, 8]

    total_recall = {k: 0 for k in k_list}
    processed_ids = set()

    #rpm: 500
    #tpm: 10,000 

    
    for idx in range(len(data)):
        idx, recall, retrieved_passages, thoughts, it = process_sample(idx, data[idx], dataset, top_k, k_list,max_steps, few_shot_samples, corpus, retriever, client, processed_ids) 
        # print metrics
        for k in k_list:
            total_recall[k] += recall[k]
            print(f'R@{k}: {total_recall[k] / (idx + 1):.4f} ', end='')
        print()
        if max_steps > 1:
            print('[ITERATION]', it, '[PASSAGE]', len(retrieved_passages), '[THOUGHT]', thoughts)
        
        # record results
        results[idx]['retrieved'] = retrieved_passages
        results[idx]['recall'] = recall
        results[idx]['thoughts'] = thoughts
            
        # if idx % 50 == 0:
        #     f = open(output_path, 'w')
        #     json.dump(results, f)
        #     f.close()

    # save results
    f = open(output_path, 'w')
    json.dump(results, f)
    f.close()
    print(f'Saved results to {output_path}')
    for k in k_list:
        #average recall (across 1,000 questions for musique)
        print(f'R@{k}: {total_recall[k] / len(data):.4f} ', end='')


