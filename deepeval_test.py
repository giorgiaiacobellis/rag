from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval import evaluate
import json
import transformers
from deepeval.models.base_model import DeepEvalBaseLLM
from transformers import AutoModelForCausalLM, AutoTokenizer

def costruisci_testcases(file):
    test_cases = []
    with open(file, 'r') as f:
        data = json.load(f)

    for question, answer, context, ground, in zip(data['data']['question'], data['data']['answer'], data['data']['contexts'], data['data']['ground_truth']):
        test_case = LLMTestCase(input=question, actual_output=answer, expected_output = ground, retrieval_context=context)
        test_cases.append(test_case)

    return test_cases


class Zephyr(DeepEvalBaseLLM):
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device="cuda:0",
            max_length=2500,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        return pipeline(prompt)
    
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Llama-3 8B"


zephyr = Zephyr()

answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model=zephyr,
    include_reason=True
)

test_cases = costruisci_testcases("dataset_prova.json")
print(evaluate(
    test_cases=[test_cases],
    metrics=[answer_relevancy]
))