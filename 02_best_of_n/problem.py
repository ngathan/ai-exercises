import os
import csv
import argparse
import pdb

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
        return response.choices[0].message.content


def extract_answer(answer):
    ix = answer.rindex("####")
    return answer[ix + len("####"):].strip()


# Part 1 of the problem -- generate k responses
def generate_response(test_dataset, llm, k):
    """
    :param test_dataset: You are given a test_dataset which is a list of problem description
    :param llm: The LLM to be used
    :param k: number of responses to generate
    :return: a list of responses for each datapoint each containing a list of k tuples (reasoning, answer)
    """
    raise NotImplementedError()


def eval_model(response_answer, gold_answers):

    predictions = [answer for (reasoning, answer) in response_answer]
    scores = [1.0 if answer.lower().strip() == gold_answer.lower().strip() else 0.0
              for (answer, gold_answer) in zip(predictions, gold_answers)]
    return scores


# Part 2 of the problem -- do majority vote (self consistency) on these k responses
def do_majority_vote(reasoning_with_k_answers):
    """
    :param reasoning_with_k_answers: A list of datapoints where each datapoint is a list of k tuples
                                    containing (reasoning, answer)
    :return: a list of (reasoning, answer) indicating the prediction. For majority vote, it is okay to return any
            reasoning that has the same answer.
    """
    raise NotImplementedError()


# Part 3 of the problem -- define a reward model using log-prob from the LLM judge
class LLMJudgeAsReward:

    def __init__(self):
        pass

    def get_reward(self, chat):
        """
        - Compute following score: given an assistant response, ask if it is a right response
        - compute   P(yes | response) and P(no | response) and then
        - compute a reward score P(yes | response) / {P(yes|response) + P(no|response)}
        :param chat: a chat consisting of the problem and the assistant response
        :return: a single reward score
        """
        raise NotImplementedError()


# Part 4 of the problem -- do best-of-n prediction using the reward model
def best_of_n(reasoning_with_k_answers, reward_model):
    """
    :param reasoning_with_k_answers: A list of datapoints where each datapoint is a list of k tuples
                                    containing (reasoning, answer)
    :param reward_model: A reward model of type LLMJudgeAsReward
    :return: a list of (reasoning, answer) indicating the prediction with the highest reward.
    """
    raise NotImplementedError()


def main(args):

    # Load the GSM8k dataset. This is a set of school math puzzle problems where given a math problem in text
    # you have to generate an answer. https://huggingface.co/datasets/openai/gsm8k
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    pdb.set_trace()
    test_dataset = [(dp["question"], extract_answer(dp["answer"])) for dp in dataset["test"]]
    test_dataset = test_dataset[:args.size]
    gold_answers = [predicted_answer for (_, predicted_answer) in test_dataset]

    # Step 1: Generate k-responses and check the answer
    llm = LLM(model=args.model)
    reasoning_with_k_answers = generate_response(test_dataset, llm, k=args.k)

    # Step 2: Compute majority vote predictions using self-consistency
    self_consistent_prediction = do_majority_vote(reasoning_with_k_answers)

    self_consistent_scores = eval_model(self_consistent_prediction, gold_answers)
    self_consistent_acc = (sum(self_consistent_scores) * 100.0) / float(len(self_consistent_scores))
    print(f"Accuracy of the LLM agent using self-consistency is {self_consistent_acc:.2f}")

    # Step 3: Define a reward model using an LLM judge
    reward_model = LLMJudgeAsReward()

    # Step 4: Do best-of-n prediction
    best_of_n_prediction = best_of_n(reasoning_with_k_answers, reward_model)

    best_of_n_scores = eval_model(best_of_n_prediction, gold_answers)
    best_of_n_acc = (sum(best_of_n_scores) * 100.0) / float(len(best_of_n_scores))
    print(f"Accuracy of the LLM agent using best-of-n is {best_of_n_acc:.2f}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="A simple program to greet a user and optionally print their age.")

    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Name of OpenAI model.")
    parser.add_argument("--size", type=int, default=50, help="Number of examples to use.")
    parser.add_argument("--k", type=int, default=8, help="Number of responses to generate per prompt.")

    args = parser.parse_args()

    main(args)
