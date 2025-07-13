# LLM Judge 

Problem Statement: Given the GSM8k dataset, generate the answer to the questions, 
use an LLM judge to evaluate the response, 
then use a self-reflection step to regenerate answers to those questions 
which the LLM judge has decided hat it's not correct.

## Installation

The default llms for this project are OpenAI models. You will need an OpenAI API Key to use them. 

Install the requirements file

```python3 -m pip install -r requirements.txt```

Before running any script, set the API key as an environment variable:

```export OPENAI_KEY=...```

## List of Exercises 

### Exercise 1: Implementing LLM Judge and Self-Reflection

![img](media/00_llm_judge.png)

The goal of this exercise is to build an LLM judge to detect failures in a model's reasoning and a self-reflection step to try to recover from those failures. Using these two steps together can help improve the model's original peformance.

The folder 00_llm_judge folder contains this exercise and contains the problem definition in `llm_judge_problem.py` and solution in `llm_judge_answers.py`
