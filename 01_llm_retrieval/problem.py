import os
import pdb
import random


from openai import OpenAI
from datasets import load_dataset


class LLM:

    def __init__(self, model):
        self.llm = OpenAI(api_key=os.getenv("OPENAI_KEY"))
        self.model = model

    def response(self, chat, temp=0, max_tokens=500):
        response = self.llm.chat.completions.create(messasges=chat,
                                                    model=self.model,
                                                    temperature=temp,
                                                    max_tokens=max_tokens)
        return response.choices[0].message


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
        assert statute["statute_idx"] not in statutes_map
        statutes_map[statute["statute_idx"]] = statute

    dataset = []

    for dp in raw_dataset:
        answer = dp["answer"]
        assert answer in ["Yes", "No"]

        processed_dp = {
            "question": dp["question"],
            "answer": answer,
            "statutes": dp["statutes"]
        }
        dataset.append(processed_dp)

    # Keep only those datasets with a single statute
    dataset = [dp for dp in dataset if len(dp["statutes"]) == 1]
    dataset = dataset[:20]

    dataset_statute = []

    for dp in dataset:

        statute_ixs = [dp["statutes"][0]["idx"]]

        # Sample false statutes
        false_statutes = random.sample(statutes_map.keys(), k=5)
        statute_ixs += [statute["idx"] for statute in false_statutes]

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
            "statutes": statutes,
            "right_statute": 0
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
        else:
            prompt = """
            Consider statutory law for {state} in the year 2021. Answer the question below.

            Question ##################

            {question}
            Answer "Yes" or "No".
            Answer:
            """

        # Complete this part
        raise NotImplementedError()

        prediction = None

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
    # for cosine, use embedding and then cosine similarity -- you can use any embedding model
    # Complete this part
    raise NotImplementedError()


def eval_retrieval(dataset, retrieval_type="ngram"):
    acc = 0
    # You are given a list of statutes, you need to predict the right one using some retrieval score
    for dp in dataset:
        question = dp["question"]
        statutes = dp["statutes"]

        predicted_ix = predicted_retrieval(question, statutes)

        if predicted_ix == 0:
            acc += 1

    acc = (acc * 100.0) / float(len(dataset))

    return acc


def eval_full_pipeline(dataset):
    acc = 0
    for dp in dataset:
        question = dp["question"]
        statutes = dp["statutes"]

        predicted_ix = predicted_retrieval(question, statutes)
        predicted_statute = statutes[predicted_ix]

        # Complete this part
        raise NotImplementedError()


def main():
    # Generate dataset
    dataset, dataset_statute = create_dataset()

    llm = LLM(model="TODO")

    # Evaluate LLM with and without statues
    eval_score_wo_statute = eval_llm(llm, dataset, use_statute=False)
    eval_score = eval_llm(llm, dataset, use_statute=True)

    print(f"LLM accuracy without using statute is {eval_score_wo_statute}%")
    print(f"LLM accuracy using statute is {eval_score}%")

    # Select the best statues for each datapoint from a list using a simple n-gram heuristic
    eval_retrieval(dataset_statute, retrieval_type="ngram")

    # Select the best statues for each datapoint from a list using vector similarity
    eval_retrieval(dataset_statute, retrieval_type="cosine")

    # Putting it together, retrieva and then answer
    eval_full_pipeline(dataset_statute)

    pdb.set_trace()