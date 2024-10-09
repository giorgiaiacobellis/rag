import subprocess
import sys

def main():
    if len(sys.argv) < 2:
        print("Uso: python runner.py <valore_variabile>")
        sys.exit(1)

    var = sys.argv[1]

    # Eseguire file1.py
    subprocess.run(['python3', 'answer_relevance.py', var])

    # Eseguire file2.py
    subprocess.run(['python3', 'faithfulness.py', var])

    # Eseguire file3.py
    subprocess.run(['python3', 'answer_similarity.py', var])

if __name__ == "__main__":
    main()
