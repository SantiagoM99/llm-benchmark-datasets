from openai import AzureOpenAI
import csv
import datetime
import json
import time

from model_connections import OpenAIConnection

"""
Usage example:

manager = AzureBatchClient()

queries = [
    (1, "What is the capital of France?"),
    (2, "Explain quantum entanglement"),
    (3, "Translate 'hola' to English")
]
prompt = "You are a helpful assistant."

manager.run_batch(
    queries=queries,
    output_csv="results/results.csv",
    system_message=prompt
)

"""

class AzureBatchClient:
    """
    Handles batch operations using an OpenAIConnection instance.
    """

    def __init__(self):

        self.connection = OpenAIConnection(use_batch=True)
        self.client: AzureOpenAI = self.connection.client
        self.model_name = self.connection.model

    def create_file(self, file_path: str) -> str:
        """
        Uploads a file to the Azure OpenAI service and returns the file ID.
       
        Parameters:
        client (AzureOpenAI): An instance of AzureOpenAI client.
        file_path (str): The path to the file to be uploaded.
       
        Returns:
        str: The ID of the uploaded file.
        """
        file = self.client.files.create(
            file=open(file_path, "rb"),
            purpose="batch"
        )
        print(file.model_dump_json(indent=2))
        return file.id

    def create_batch_job(self, file_id: str) -> str:
        """
        Creates a batch job using the uploaded file and returns the batch job ID.
       
        Parameters:
        client (AzureOpenAI): An instance of AzureOpenAI client.
        file_id (str): The ID of the uploaded file.
       
        Returns:
        str: The ID of the created batch job.
        """
        batch_response = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/chat/completions",
            completion_window="24h",
        )
        print(batch_response.model_dump_json(indent=2))
        return batch_response.id

    def validate_output_file(self, batch_id: str) -> str:
        """
        Retrieves the status of a batch job and returns the output file ID if available.
       
        Parameters:
        client (AzureOpenAI): An instance of AzureOpenAI client.
        batch_id (str): The ID of the batch job.
       
        Returns:
        str or None: The ID of the output file if the batch is completed; otherwise, None.
        """
        batch_response = self.client.batches.retrieve(batch_id)
        status = batch_response.status
        print(f"{datetime.datetime.now()} Batch Id: {batch_id}, Status: {status}")
       
        if batch_response.status == "failed":
            for error in batch_response.errors.data:
                print(f"Error code {error.code} Message {error.message}")
            return None
        elif batch_response.status == "completed":
            print("File ready")
            return batch_response.output_file_id
        else:
            print("File not ready... wait 60s more before sending the request again")
            return None

    def list_batches(self):
        """
        Retrieves and prints the list of current batch jobs along with their status.
       
        Parameters:
        client (AzureOpenAI): An instance of AzureOpenAI client.
        """
        list_batch = self.client.batches.list().data
        print(f"==== Current amount of batch jobs: {len(list_batch)} ======")
        for batch in list_batch:
            print(f"Batch id: {batch.id}, completed_jobs: {batch.request_counts.completed}, "
                  f"failed_jobs: {batch.request_counts.failed}, total_jobs: {batch.request_counts.total}")

    def process_answer_file(self, output_file_id: str, csv_filename: str):
        """
        Processes the output file from a batch job and writes the results to a CSV file.
       
        Parameters:
        client (AzureOpenAI): An instance of AzureOpenAI client.
        output_file_id (str): The ID of the output file.
        csv_filename (str): The name of the CSV file to write the results to.
        """
        if output_file_id:
            with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                # Write header
                writer.writerow(["Task ID", "Message", "Total Tokens"])
                   
                file_response = self.client.files.content(output_file_id)
                raw_responses = file_response.text.strip().split('\n')

                for raw_response in raw_responses:
                    data = json.loads(raw_response)
                    task_id = data.get("custom_id", "")
                    message = data["response"]["body"]["choices"][0]["message"]["content"]
                    total_tokens = data["response"]["body"]["usage"]["total_tokens"]
                    writer.writerow([task_id, message, total_tokens])

    @staticmethod
    def generate_jsonl(file_path, model_name, queries, system_message=None):
        """
        Generate a JSONL file with structured chat completion requests.
       
        :param file_path: Path to save the JSONL file.
        :param model_name: Name of the AI model to be used.
        :param queries: List of user queries with the id: (id, query).
        :param system_message: Optional system message; if None, it is omitted.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            for id, query in queries:
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
               
                messages.append({"role": "user", "content": query})
               
                data = {
                    "custom_id": f"task-{id}",
                    "method": "POST",
                    "url": "/chat/completions",
                    "body": {
                        "model": model_name,
                        "messages": messages
                    }
                }

                f.write(json.dumps(data) + "\n")
    def run_batch(self, queries, output_csv, jsonl_path="batch_input.jsonl",
                  system_message=None, wait_seconds=30):
        """
        Executes the full batch workflow: generate JSONL, upload file,
        create batch job, poll until completion, and process output CSV.

        Parameters:
        queries (list): List of tuples (id, query).
        output_csv (str): Path for the generated CSV with results.
        jsonl_path (str): Where to store the temporary JSONL batch file.
        system_message (str): Optional system message for each chat request.
        wait_seconds (int): Seconds to wait between status checks.

        Returns:
        str: Path to the final CSV file.
        """
        
        # 1. Generate JSONL
        print("Generating JSONL...")
        self.generate_jsonl(
            file_path=jsonl_path,
            model_name=self.model_name,
            queries=queries,
            system_message=system_message
        )

        # 2. Upload file
        print("Uploading file to Azure...")
        file_id = self.create_file(jsonl_path)

        # 3. Create batch job
        print("Creating batch job...")
        batch_id = self.create_batch_job(file_id)

        # 4. Poll status
        print(f"Waiting for batch to complete... Checking every {wait_seconds} seconds.")
        output_file_id = None

        while output_file_id is None:
            output_file_id = self.validate_output_file(batch_id)
            if output_file_id is None:
                time.sleep(wait_seconds)

        # 5. Process the answer file
        print("Processing output file...")
        self.process_answer_file(output_file_id, output_csv)

        print(f"Batch completed successfully. Results saved to {output_csv}")
        return output_csv

