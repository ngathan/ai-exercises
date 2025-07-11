import os
import csv

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


def extract_answer(answer):
    ix = answer.rindex("####")
    return answer[ix + len("####"):].strip()


# Part 1 of the problem -- generate
def generate_response(test_dataset, llm):
    """
    :param test_dataset: You are given a test_dataset which is a list of problem description
    :param llm: The LLM to be used
    :return: a list of (reasoning, answer)
    """
    raise NotImplementedError()


def eval_model(response_answer, gold_answers):

    predictions = [answer for (reasoning, answer) in response_answer]
    scores = [1.0 if answer.lower().strip() == gold_answer.lower().strip() else 0.0
              for (answer, gold_answer) in zip(predictions, gold_answers)]
    return scores


# Part 2 of the problem
def generate_judge_response(test_dataset_with_response):
    """
    :param test_dataset_with_response: You are given the train_dataset which is a list of tuple (problem description, llm reasoning)
    :return: a list of (judge_response, judge's answer which is True or Wrong)
    """
    raise NotImplementedError()


# Part 3 of the problem -- evaluate the LLM judge
def eval_llm_judge(predictions, gold_eval):
    """
    :param predictions: A list of 0-1 score whether a given prediction is right (1) or wrong (0)
    :param gold_eval: A list of ground truth scores whether a given prediction is right (1) or wrong (0)
    :return: overall
    """
    raise NotImplementedError()


# Part 4 of the problem -- self-reflection if the LLM judge produced a wrong answer
def self_reflection(history):
    """
    :param history: A single datapoint's history of original question, model's first reasoning, and llm judge response
    :return: output both the final reasoning and the extracted final answer
    """
    raise NotImplementedError()


def save(histories):

    fieldnames = ["question", "model_reasoning", "judge_reasoning", "reflection_reasoning"]
    with open("./history.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for history in histories:
            row = {k: v for (k, v) in zip(fieldnames, history)}
            writer.writerow(row)

def main():

    # Load the GSM8k dataset. This is a set of school math puzzle problems where given a math problem in text
    # you have to generate an answer.
    # https://huggingface.co/datasets/openai/gsm8k
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    test_dataset = [(dp["question"], extract_answer(dp["answer"])) for dp in dataset["test"]]
    test_dataset = test_dataset[:20]
    gold_answers = [predicted_answer for (_, predicted_answer) in test_dataset]

    # Step 1: Generate responses and check the answer
    llm = LLM(model="")
    reasoning_with_answer = generate_response(test_dataset, llm)

    model_scores = eval_model(reasoning_with_answer, gold_answers)
    model_acc = (sum(model_scores) * 100.0) / float(len(model_scores))
    print(f"Accuracy of the LLM agent is {model_acc:.2f}")

    # Step 2: Generate LLM judge response
    test_dataset_with_response = [(dp["question"], reasoning_with_answer_[0])
                                  for reasoning_with_answer_, dp in zip(reasoning_with_answer, test_dataset)]
    judge_response_with_answer = generate_judge_response(test_dataset_with_response)
    judge_answers = [judge_answer for (judge_response, judge_answer) in judge_response_with_answer]

    # Step 3: Evaluate LLM judge response
    # - judge_predictions are 0/1 whether the LLM judge thinks the datapoint was answered correctly or not
    # - model_scores contains 0/1 score on whether each datapoint was answered correctly or not
    # use these two to evaluate the LLM judge
    judge_scores = eval_llm_judge(judge_answers, model_scores)
    judge_acc = (sum(judge_scores) * 100.0) / float(len(model_scores))
    print(f"Accuracy of the LLM Judge is {judge_acc:.2f}")

    # Step 4: for those datapoints where the LLM judge predicted the model's response to be wrong
    # now prompt the model to recompute their response given the original prompt, model's initial response, and
    # judge's predictions all put in a chat history.

    final_scores = []
    full_histories = [] # for debugging
    for (dp, reasoning_with_answer_, judge_response_with_answer_) in \
            zip(test_dataset, reasoning_with_answer, judge_response_with_answer):

        model_reasoning, model_answer = reasoning_with_answer_
        judge_reasoning, judge_answer = judge_response_with_answer_

        # History of this datapoint containing the first question, model's reasoning and then the judge's reasoning
        history = [dp["question"], model_reasoning, judge_reasoning]

        if judge_answer == 1:  # Correct
            final_answer = model_answer
            history.append("N/A")   # N/A indicating no self-reflection reasoning
        else:
            # Self-reflection step: LLM judge thinks we made a mistake answering this datapoint, so retry
            reflection_reasoning, final_answer = self_reflection(history)
            history.append(reflection_reasoning)

        final_score = 1.0 if dp["answer"].lower().strip() == final_answer.lower().strip() else 0
        final_scores.append(final_score)
        full_histories.append(history)

    final_acc = (sum(final_scores) * 100.0) / float(len(final_scores))
    print(f"Final Accuracyy of the Self-Reflection LLM Agent is {final_acc:.2f}")

    save(full_histories)

    if final_acc > model_acc:
        print("You were able to achieve higher final accuracy using self-reflection than the original model.")


if __name__ == '__main__':
    main()