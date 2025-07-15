# Retrieval Exercise  

**Problem Statement:** Given the GSM8k dataset, generate the answer to the questions, 
use an LLM judge to evaluate the responses, 
then use a self-reflection step to regenerate answers to those questions 
which the LLM judge has decided that they were not correct.
Compare the accuracy of each step to record whether the AI system is better than simply using the LLM generator.

**AI system design:**

![img](../media/00_llm_judge.png)

**Hints:**

Hint 1: When building the pipeline experimenting with the first 20, 50, 100, 200 data points in the dataset to see whether the pipeline works as expected
Hint 2 (Advanced level): Beyond TFIDF and ngram in the problem statement, you can experiment with more advanced embedding methodology such as embedding models from OpenAI, Anthropic.  
Hint 3 (Advanced level): Implement an index using FAISS to store all the statutes to speed up the cosine similarities calculation for the entire dataset  

