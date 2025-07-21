# LLM Judge 

**Problem Statement:** Given the GSM8k dataset, generate the answer to the questions, 
use an LLM judge to evaluate the responses, 
then use a self-reflection step to regenerate answers to those questions 
which the LLM judge has decided that they were not correct.
Compare the accuracy of each step to record whether the AI system is better than simply using the LLM generator.

**AI system design:**

![img](../media/00_llm_judge.png)

**Hints:**

Hint 1: When prompting the LLM at the judge and self-reflection steps, keep adding the assistant's responses to the chat.

**Real world LLM-as-judge examples** 

[Improving retrieval with LLM-as-a-judge](https://blog.vespa.ai/improving-retrieval-with-llm-as-a-judge/)

**Notes** 
Part of the GMS8k training dataset was included in GPT-4 pretraining mix to improve the model's ability to do mathematical reasoning [GPT4 Technical Report](https://arxiv.org/abs/2303.08774). However, in this exercise, we only use the test set. This still doesn't necessarily prevent some data leakage as argued in [AI Engineering](https://www.amazon.com/AI-Engineering-Building-Applications-Foundation/dp/1098166302/ref=sr_1_3?crid=1BCWKREW6UN5E&dib=eyJ2IjoiMSJ9.29d3zXlbjkjfzj-S1rS3rOF_sXH0xThYce2wTyB3xFGSHJFABS2yEyR8ePj8NCxCi1ULjVhah_LtIcaR041qNFUF2B-oNtowYb2E-HxHrI1Wvq95-ApfkA3u7Ma5s-FZJGsOUKaXTPnMrqgdj6gJv6oil9kO4ytH5MQzEkG_Kl4pUt-hzjNri8SgFyFt5ge05WRlQqrreixkZNeTZ_52hHW0h_d3Q2gU_RQd9IF0jKI.9QKRYWH74x2MEwR-ohELsQMUf6ZFCQlrEpf0ML8sxK0&dib_tag=se&keywords=ai+engineer&qid=1752858045&sprefix=ai+engine%2Caps%2C275&sr=8-3) by Chip Huyen. For more information about training data of large language models, you can read more of chapter 2 Understanding Foundation Models, specifically the Training data part.
