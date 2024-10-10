import subprocess
import sys
import subprocess
import sys

def main():
    if len(sys.argv) < 2:
        print("Uso: python runner.py <file1> <file2> ... <fileN>")
        sys.exit(1)

    # Itera attraverso tutti i file passati come argomenti
    for var in sys.argv[1:]:
        print(f"Eseguendo script sui dati del file: {var}")


        # Eseguire answer_relevance.py
        subprocess.run(['python3', 'answer_relevance.py', var])

        # Eseguire faithfulness.py
        subprocess.run(['python3', 'faithfulness.py', var])

        # Eseguire answer_similarity.py
        subprocess.run(['python3', 'answer_similarity.py', var])

        # Eseguire answer_correctness.py
        subprocess.run(['python3', 'answer_correctness.py', var])

        # Eseguire context_relevance.py
        subprocess.run(['python3', 'context_relevance.py', var])

if __name__ == "__main__":
    main()


'''
 python3 runner.py dataset_2024-10-07_21-24-15.json dataset_llama_0505_stella.json dataset_2024-10-07_21-32-02.json dataset_llama_0808_stella.json dataset_Llama-0202.json dataset_llama_11_stella.json mist05mp.json dataset_Llama-0505.json dataset_mistral_0202_stella.json mistral_103_stella.json dataset_Llama-0808.json dataset_mistral_0208_stella.json mist03mp.json dataset_Llama-11.json dataset_mistral_0505_stella.json dataset_Mistral-11.json dataset_mistral_0808_stella.json dataset_mistral_11.json dataset_2024-09-27_18-23-43.json dataset_zephyr_0202_stella.json results_2024-09-27_18-26-12.json dataset_2024-10-01_16-00-48.json dataset_gemma_0202_mpnet.json dataset_zephyr_0208_stella.json dataset_2024-10-05_20-02-33.json dataset_gemma_0202_stella.json dataset_zephyr_0505_stella.json dataset_2024-10-06_11-42-12.json dataset_gemma_0208_mpnet.json dataset_zephyr_0808_stella.json dataset_2024-10-06_18-04-50.json dataset_gemma_0208_stella.json dataset_zephyr_11_stella.json dataset_2024-10-06_19-21-42.json dataset_gemma_0505_mpnet.json dataset_2024-10-06_19-50-49.json dataset_gemma_0505_stella.json dataset_2024-10-06_20-06-46.json dataset_gemma_0808.json gemma03mp.json zep03mp.json gemma03stella.json zep03stella.json zep05mp.json llama03mp.json zep05stella.json llama03stella.json llama05mp.json
 
 '''   