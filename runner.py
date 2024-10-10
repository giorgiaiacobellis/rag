import subprocess
import sys

'''def main():
    if len(sys.argv) < 2:
        print("Uso: python runner.py <valore_variabile>")
        sys.exit(1)

    var = sys.argv[1]

    # Eseguire file1.py
    subprocess.run(['python3', 'langchain_pipeline.py', var])

    # Eseguire file1.py
    subprocess.run(['python3', 'answer_relevance.py', var])

    # Eseguire file2.py
    subprocess.run(['python3', 'faithfulness.py', var])

    # Eseguire file3.py
    subprocess.run(['python3', 'answer_similarity.py', var])

        # Eseguire file3.py
    subprocess.run(['python3', 'answer_correctness.py', var])

        # Eseguire file3.py
    subprocess.run(['python3', 'context_relevance.py', var])

if __name__ == "__main__":
    main()
'''
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
        #subprocess.run(['python3', 'answer_relevance.py', var])

        # Eseguire faithfulness.py
        #subprocess.run(['python3', 'faithfulness.py', var])

        # Eseguire answer_similarity.py
        #subprocess.run(['python3', 'answer_similarity.py', var])

        # Eseguire answer_correctness.py
        subprocess.run(['python3', 'answer_correctness.py', var])

        # Eseguire context_relevance.py
        subprocess.run(['python3', 'context_relevance.py', var])

if __name__ == "__main__":
    main()