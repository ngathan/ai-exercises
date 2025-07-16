import os
import pdb
import random

from openai import OpenAI
from datasets import load_dataset

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer




class LLM:

    def __init__(self, model):
        self.llm = OpenAI(api_key=os.getenv("OPENAI_KEY"))
        self.model = model

    def response(self, chat, temp=0, max_tokens=500):

        response = self.llm.chat.completions.create(model=self.model,
                                                        messages=chat,
                                                        temperature=temp,
                                                        max_tokens=max_tokens)
        return response.choices[0].message.content


def create_dataset():

    # Load questions -- list of
    #           ['idx', 'state', 'question', 'answer', 'question_group', 'statutes', 'original_question', 'caveats']
    #           statute_idx itself has statute_idx, citation, excerpt
    raw_dataset = load_dataset("reglab/housing_qa", "questions", split="test")

    # Load questions_aux
    # questions_aux = load_dataset("reglab/housing_qa", "questions_aux", split="test")

    # Load statutes -- list of ['citation', 'path', 'state', 'text', 'idx']
    statutes = load_dataset("reglab/housing_qa", "statutes", split="corpus")


    statutes_map = {}

    for statute in statutes:
        assert statute["idx"] not in statutes_map ## this will give key error because statute_idx is not there
        statutes_map[statute["idx"]] = statute

    dataset = []

    for dp in raw_dataset:
        answer = dp["answer"]
        assert answer in ["Yes", "No"]

        processed_dp = {
            "question": dp["question"],
            "answer": answer,
            "state": dp["state"],
            "statutes": dp["statutes"]
        }
        dataset.append(processed_dp)

    # Keep only those datasets with a single statute for easy retrieval
    dataset = [dp for dp in dataset if len(dp["statutes"]) == 1] # total:  2988

    # only experiment with the first 20, 50, 100, 200 documents for quick results, then optimize with vector database later at the retrieval step
    dataset = dataset[:1000]

    dataset_statute = []


    for dp in dataset:

        statute_ixs = [dp["statutes"][0]["statute_idx"]]
        # Sample false statutes

        false_statutes = random.sample(list(statutes_map.keys()), k=5)
        statute_ixs += [statute in false_statutes]
        statutes = []

        for statute_ix in statute_ixs:
            statute = statutes_map[statute_ix]
            statutes.append({
                    "citation": statute["citation"],
                    "text": statute["text"]
            })

        dataset_statute.append({
            "question": dp["question"],
            "answer": dp["answer"],
            'state': dp['state'],
            "statutes": statutes, # these statutes have 6 answers so to say, and one of it is correct
            "right_statute": 0 ## right statute index is number 0
        })

    return dataset, dataset_statute


def eval_llm(llm, dataset, use_statute=True):

    acc = 0.0
    for dp in dataset:

        if use_statute:
            prompt = """
            Consider statutory law for {state} in the year 2021. Read the following statute excerpts which govern housing law in this state, and answer the question below.
    
            Statutes ################## 
    
            {statute_list}
    
            Question ##################
    
            {question}
            Answer "Yes" or "No".
            Answer:
            """

            prompt = prompt.format(state=dp['state'], statute_list=dp['statutes'], question=dp['question'])

        else:
            prompt = """
            Consider statutory law for {state} in the year 2021. Answer the question below.
            
            Question ##################

            {question}
            Answer "Yes" or "No".
            Answer:
            """

            prompt = prompt.format(state=dp['state'], question=dp['question'])

        #Complete this part
        #raise NotImplementedError()

        chat = [{"role": "system", "content": prompt}]
        response = llm.response(chat=chat, temp=0, max_tokens=500)

        prediction = response  # extract the answer from response

        if prediction == dp["answer"]:
            acc += 1

    acc = (acc * 100.0) / float(len(dataset))

    return acc


def predicted_retrieval(question, statutes, retrieval_type="ngram"):
    """
    :param question:
    :param statutes:
    :param retrieval_type:
    :return: index of the statute in the list which is the chosen one
    """
    assert retrieval_type in ["ngram", "cosine"]
    # for ngram, use some ngram style similarity

    if retrieval_type=='ngram':
        n = 10 # change this to reflect the similarity between question and statutes
        similarities = []
        for statute in statutes:
            vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
            X = vectorizer.fit_transform([question, statute['text']])
            similarity = cosine_similarity(X[0:1], X[1:2])[0][0]
            similarities.append(similarity)

        max_similarity = max(similarities)

        # for cosine, use embedding and then cosine similarity -- you can use any embedding model
        # # Complete this part
        #pdb.set_trace()

        return similarities.index(max_similarity)

    if retrieval_type=='cosine':
       # using simple TFIDF embedding model

        similarities = []
        for statute in statutes:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([question, statute['text']])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            similarities.append(similarity)

        max_similarity = max(similarities)
        return similarities.index(max_similarity)


def eval_retrieval(dataset, retrieval_type="ngram"):

    acc = 0
    # You are given a list of statutes, you need to predict the right one using some retrieval score
    for dp in dataset:
        question = dp["question"]
        statutes = dp["statutes"]

        predicted_ix = predicted_retrieval(question, statutes, retrieval_type=retrieval_type)

        if predicted_ix == 0:
            acc += 1

    acc = (acc * 100.0) / float(len(dataset))

    return acc


def eval_full_pipeline(dataset, llm):

    acc = 0
    for dp in dataset:

        question = dp["question"]
        statutes = dp["statutes"]

        predicted_ix = predicted_retrieval(question, statutes, retrieval_type='cosine')
        predicted_statute = statutes[predicted_ix]
        prompt = """
                  Consider statutory law for {state} in the year 2021. Read the following statute excerpts which govern housing law in this state, and answer the question below.

                  Statutes ################## 

                  {statute_list}

                  Question ##################

                  {question}
                  Answer "Yes" or "No".
                  Answer:
                  """

        prompt = prompt.format(state=dp['state'], statute_list=predicted_statute, question=dp['question'])

        chat = [{"role": "system", "content": prompt}]
        response = llm.response(chat=chat, temp=0, max_tokens=500)
        prediction = response  # extract the answer from response

        if prediction == dp["answer"]:
            acc += 1

    acc = (acc * 100.0) / float(len(dataset))

    return acc


def main():

    # Generate dataset
    dataset, dataset_statute = create_dataset()

    llm = LLM(model="gpt-4.1-nano-2025-04-14") ## use the mano model for cost saving
    # Evaluate LLM with and without statues
    eval_score_wo_statute = eval_llm(llm, dataset, use_statute=False)
    eval_score = eval_llm(llm, dataset, use_statute=True)

    print(f"LLM accuracy without using statute is {eval_score_wo_statute}%")
    print(f"LLM accuracy using statute is {eval_score}%")

    # Select the best statues for each datapoint from a list using a simple n-gram heuristic
    #eval_retrieval(dataset_statute, retrieval_type="ngram")

    # Select the best statues for each datapoint from a list using vector similarity
    cosine_retrieval_acc = eval_retrieval(dataset_statute, retrieval_type="cosine")
    print(f"Accuracy using tfidf embedding is {cosine_retrieval_acc}")


    # Putting it together, retrieval and then answer
    final_score = eval_full_pipeline(dataset_statute, llm)

    print(f"Final accuracy with retrieval and llm is:  {final_score}")


if __name__ == '__main__':
    main()