import os
import re
import csv
import pdb

from openai import OpenAI
from datasets import load_dataset

math_answer_prompt = """You are solving high school math problems. 
        Read the problem statement. Think step by step. Then devise a consise solution to solve the math question. 
        When answer, first provide the reasoning, or the rationale of how the to solve the problem.
        At the end answer following the format "Final answer is" and the final numerical number. 
        Remove any sign such as dollar sign ($) before the final numerical number. 
        Remove words such as approximate, about before the final numerical number. 
        Remove any unit words such as mile, hours, kg, lb, pound after the final numerical number.  
        Rememove signs such as **, dot (.), comma (,) after the final numerical answer.  
        Example: Final answer is 8 \n.
        """


class LLM:

    def __init__(self, model):
        self.llm = OpenAI(api_key=os.getenv("OPENAI_KEY"))
        self.model = model
    def response(self, chat, temp=0, max_tokens=700):

        response = self.llm.chat.completions.create(model=self.model,
                                                        messages=chat,
                                                        temperature=temp,
                                                        max_tokens=max_tokens)

        return response.choices[0].message.content


def extract_answer(answer):
    ix = answer.rindex("####")
    ans = answer[ix + len("####"):].strip()
    return ans

def extract_answer_llm(answer):
    if "Final answer is" in answer:
        ix = answer.rindex("Final answer is")
        ans = answer[ix + len("Final answer is"):].strip().rstrip('.')
    else:
        print("something wrong, find the answer")
        ans = answer
        print(answer)

    if ans.startswith('$'):
        ans = ans[1:].strip().rstrip('.')

    numbers = re.findall(r'\d+(?:\.\d+)?', ans)
    return numbers[0] ## assuming that there is only on valid number in the final answer

def extract_reasoning(answer):
    if "final answer is" in answer.lower():
        ix = answer.lower().rindex("final answer is")
        return answer[:ix].strip()


# Step 1 of the system: Generate
def generate_response(test_dataset, llm):
    """
    :param test_dataset: You are given a test_dataset which is a list of problem description
    :param llm: The LLM to be used
    :return: a list of (reasoning, answer)
    """
    ans = []
    for i, (question, _) in enumerate(test_dataset):
        if i % 200 == 0:
            print('processing question ', i)

        chat = [
        {"role": "system", "content": math_answer_prompt},
        {"role": "user", "content": question}
        ]

        response = llm.response(chat=chat, temp=0, max_tokens=1000)
        answer = extract_answer_llm(response)
        reasoning = response
        ans.append((reasoning, answer))

    return ans



def eval_model(response_answer, gold_answers):

    predictions = [answer for (reasoning, answer) in response_answer]
    scores = [1.0 if answer.lower().strip() == gold_answer.lower().strip() else 0.0 for (answer, gold_answer) in zip(predictions, gold_answers)]
    return scores


# Part 2 of the  system: Using an LLM judge to evaluate the previous generation
def generate_judge_response(test_dataset_with_response, llm):
    """
    :param test_dataset_with_response: You are given the train_dataset which is a list of tuple (problem description, llm reasoning)
    :return: a list of (judge_response, judge's answer which is True or Wrong)
    """

    ans = []
    for i, (question, answer) in enumerate(test_dataset_with_response):
        if i % 200 == 0:
            print('processing question ', i)

        chat = [
        {"role": "system", "content": math_answer_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
        {"role": "user", "content": "Is the provided solution correct or not? Check the reasoning and if there is any doubt then call it incorrect. "
                                    "Please put your answer inside <answer></answer> tag and only answer 'correct' or 'wrong'"}
        ]

        response = llm.response(chat=chat, temp=0.6, max_tokens=1000)
        reasoning = response

        if "<answer>" in response:
            response = response[response.index("<answer>") + len("<answer>"):]
        if "</answer>" in response:
            response = response[:response.index("</answer>")]

        if "correct" in response:
            answer = True
        else:
            answer = False

        ans.append((reasoning, answer))

    return ans


# Part 3 of the system -- evaluate the LLM judge
def eval_llm_judge(predictions, gold_eval):
    """
    :param predictions: A list of 0-1 score whether a given prediction is right (1) or wrong (0)
    :param gold_eval: A list of ground truth scores whether a given prediction is right (1) or wrong (0)
    :return: overall
    """
    scores = [1.0 if int(prediction) == gold_answer else 0.0 for (prediction, gold_answer) in
              zip(predictions, gold_eval)]

    return scores


# Part 4 of the problem -- self-reflection if the LLM judge said that a previous answer was wrong. Then correct that wrong answer
def self_reflection(history, llm):
    """
    :param history: A single datapoint's history of original question, model's first reasoning, and llm judge response
    :return: output both the final reasoning and the extracted final answer
    """
    question, model_reasoning, judge_reasoning = history
    chat = [
        {"role": "system", "content": math_answer_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": model_reasoning},
        {"role": "user", "content": "Is the provided solution correct or not? Check the reasoning and if there is any doubt then call it incorrect. "
                                    "Please put your answer inside <answer></answer> tag and only answer 'correct' or 'wrong'"},
        {"role": "assistant", "content": judge_reasoning},
        {"role": "user", "content": "Okay since you think the answer is wrong can you generate a better response?"}
    ]

    response = llm.response(chat=chat, temp=0, max_tokens=1000)

    answer = extract_answer_llm(response)
    reasoning = response

    return (reasoning, answer)

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
    # this is only the validation dataset
    # with split='test', the dataset will contain only the test portion
    dataset = load_dataset("openai/gsm8k", "main")

    # examining the test and answer question
    test_dataset = [(dp["question"], extract_answer(dp["answer"])) for dp in dataset["test"]]
    #test_dataset = test_dataset[:150]
    gold_answers = [predicted_answer for (_, predicted_answer) in test_dataset]

    # Step 1: Generate responses and check the answer

    llm = LLM(model="gpt-4.1-nano-2025-04-14") #using gpt-4.1-nano-2025-04-14 for this math problem to save cost
    reasoning_with_answer = generate_response(test_dataset, llm)
    model_scores = eval_model(reasoning_with_answer, gold_answers)
    model_acc = (sum(model_scores) * 100.0) / float(len(model_scores))
    print(f"Accuracy of the LLM agent is {model_acc:.2f}")

    # Step 2: Generate LLM judge response
    test_dataset_with_response = [(dp[0], reasoning_with_answer_[0])
                                  for reasoning_with_answer_, dp in zip(reasoning_with_answer, test_dataset)]


    judge_response_with_answer = generate_judge_response(test_dataset_with_response, llm)

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
        history = [dp[0], model_reasoning, judge_reasoning]

        if judge_answer == 1:  # Correct
            final_answer = model_answer
            history.append("N/A")   # N/A indicating no self-reflection reasoning
        else:

            # Self-reflection step: LLM judge thinks we made a mistake answering this datapoint, so retry
            reflection_reasoning, final_answer = self_reflection(history, llm)
            history.append(reflection_reasoning)

        final_score = 1.0 if dp[1].lower().strip() == final_answer.lower().strip() else 0
        final_scores.append(final_score)
        full_histories.append(history)

    final_acc = (sum(final_scores) * 100.0) / float(len(final_scores))
    print(f"Final Accuracy of the Self-Reflection LLM Agent is {final_acc:.2f}")

    save(full_histories)

    if final_acc > model_acc:
        print("You were able to achieve higher final accuracy using self-reflection than the original model.")


if __name__ == '__main__':
    main()
