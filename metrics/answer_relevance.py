import os
import data
import json
import sys


from langchain_community.llms.vllm import VLLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


os.environ["OPENAI_API_KEY"] = ("sk-rf-yLyTntiSYVkhQm8O5bgiGQn1GAYwlPngB80vlNsT3BlbkFJtntowM_ykl6TVjFdZalhu6MuYHeBdSMh1OJmtqbH4A")
os.environ["HUGGINGFACE_ACCESS_TOKEN"] = ("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")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]="ragTestServer"


llm = VLLM(
            model=data.config_eval["llm"]["model"],
            top_p=data.config_eval["llm"]["top_p"],
            max_new_tokens=4000,
            temperature=data.config_eval["llm"]["temperature"],
            top_k=data.config_eval["llm"]["top_k"],
            trust_remote_code= True
        )

# Function to generate question variations from the answer using LangChain VLLM
def generate_questions_from_answer(answer):
    prompt = (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"Data la seguente risposta, genera tre domande in italiano pertinenti "
        f"come se fossero domande per cui questa risposta è corretta. Rispondi solo con le tre domande generate separate da &.<|eot_id|>\n"
        f"<|start_header_id|>user<|end_header_id|>"
        f"Ecco la risposta:\n\n{answer}<|eot_id|>\n\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    # Generate the questions using the LangChain VLLM model
    result = llm.invoke(prompt)
    print(result)

    questions = result.strip().split("&")[:3]  # Assuming the model returns each question on a new line
    return questions

# Function to calculate cosine similarity
def calculate_cosine_similarity(original_question, generated_questions):
    questions = [original_question] + generated_questions
    vectorizer = TfidfVectorizer().fit_transform(questions)
    vectors = vectorizer.toarray()
    
    # Compute cosine similarity between the original question and each generated question
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    return cosine_similarities.mean()

# Main function to calculate answer relevancy from a dataset
def answer_relevancy_score(dataset_path):
    # Load the dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    total_sim = 0
    for i, item in enumerate(data['data']['question']):
        original_question = item
        answer = data['data']['answer'][i]
        
        # Generate the questions using the answer
        generated_questions = generate_questions_from_answer(answer)
        
        # Calculate cosine similarity
        avg_cosine_similarity = calculate_cosine_similarity(original_question, generated_questions)
        
        # Store the result
        results.append({
            'question': original_question,
            'generated_question_1': generated_questions[0],
            'generated_question_2': generated_questions[1],
            'generated_question_3': generated_questions[2],
            'average_cosine_similarity': avg_cosine_similarity
        })
    
        total_sim = total_sim + avg_cosine_similarity
    # Display the results
    '''
    for result in results:
        print(f"Question: {result['question']}")
        print(f"Generated Questions: {result['generated_question_1']}, {result['generated_question_2']}, {result['generated_question_3']}")
        print(f"Average Cosine Similarity: {result['average_cosine_similarity']}\n")
    
    '''
    print(f"Aanswer relavancy: {total_sim/len(data['data']['question'])}")

    return total_sim/len(data['data']['question'])

# Example usage
dataset_path = 'dataset_gemma_11_stella.json'  # Replace with your dataset file
result = answer_relevancy_score(dataset_path)


def main():
    filename = sys.argv[1]
    print(f"Test file: {filename}")
    result = answer_relevancy_score(filename)
    print(f"Answer relevancy score totale: {result}")

if __name__ == "__main__":
    main()