import os
import json
import time
import datetime

from preprocess_data_pipeline import preprocess_comments
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_statistics_dict(true_labels, predicted_labels):
    accuracy = round(accuracy_score(true_labels, predicted_labels), 3)
    precision = round(precision_score(true_labels, predicted_labels), 3)
    recall = round(recall_score(true_labels, predicted_labels), 3)
    f1 = round(f1_score(true_labels, predicted_labels), 3)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def submit_all_subreddits():
    client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

    subreddit_data = {}
    with open(
        f"./data/rule_moderation/subreddit_rules_w_description.jsonl"
    ) as srd_file:
        for line in srd_file:
            line_dict = json.loads(line)
            subreddit = line_dict["subreddit"]

            print(f"Submitting {subreddit}.jsonl")
            response = client.files.create(
                file=open(f"./preprocessed_data/{subreddit}.jsonl", "rb"),
                purpose="batch",
            )

            if response.created_at:
                print(f"Successfully submitted {response.filename}")
                subreddit_data[subreddit] = {"file_id": response.id}

    with open("./subreddit_batch_file_index.jsonl", "w") as subreddit_bfi:
        subreddit_bfi.write(json.dumps(subreddit_data))


def create_batch_for_subreddit(client, subreddit):
    with open("./subreddit_batch_file_index.jsonl", "r") as subreddit_bfi:
        subreddit_data = json.loads(subreddit_bfi.read())

    file_id = subreddit_data[subreddit]["file_id"]

    print(f"Submitting {subreddit} batch")
    response = client.batches.create(
        input_file_id=file_id, endpoint="/v1/chat/completions", completion_window="24h"
    )

    if response.id:
        print(f"Successfully submitted {subreddit} batch")
        subreddit_data[subreddit]["batch_id"] = response.id

    with open("./subreddit_batch_file_index.jsonl", "w") as subreddit_bfi:
        subreddit_bfi.write(json.dumps(subreddit_data))


def check_batch_status(client, subreddit):
    with open("./subreddit_batch_file_index.jsonl", "r") as subreddit_bfi:
        subreddit_data = json.loads(subreddit_bfi.read())

    batch = client.batches.retrieve(subreddit_data[subreddit]["batch_id"])
    status = batch.status
    completed = batch.request_counts.completed
    failed = batch.request_counts.failed
    total = batch.request_counts.total

    return {"status": status, "completed": completed, "failed": failed, "total": total}


def create_batch_result(client, subreddit):
    with open("./subreddit_batch_file_index.jsonl", "r") as subreddit_bfi:
        subreddit_data = json.loads(subreddit_bfi.read())

    response_file_id = client.batches.retrieve(
        subreddit_data[subreddit]["batch_id"]
    ).output_file_id

    content = client.files.content(response_file_id)
    with open(f"batch_results/{subreddit}.jsonl", "wb") as file:
        file.write(content.read())

    subreddit_data[subreddit]["response_file_id"] = response_file_id
    with open("./subreddit_batch_file_index.jsonl", "w") as subreddit_bfi:
        subreddit_bfi.write(json.dumps(subreddit_data))


def print_stats_for_subreddit(subreddit):
    labels = []
    predicted_labels = []

    predicted_labels_indices = []

    with open(f"./batch_results/{subreddit}.jsonl") as file:
        for line in file:
            s = json.loads(line)
            predicted_labels.append(
                json.loads(s["response"]["body"]["choices"][0]["message"]["content"])[
                    "would_moderate"
                ]
            )
            predicted_labels_indices.append(
                int(s["custom_id"][8:])
            )  # Takes out the "comment_" from "comment_n"

    predicted_labels_indices_set = set(predicted_labels_indices)
    for count, comment_dict in enumerate(preprocess_comments(subreddit)):
        if count in predicted_labels_indices_set:
            labels.append(comment_dict["label"])

    print(get_statistics_dict(labels, predicted_labels))


def start_subreddit_batch_pipeline(subreddit):
    client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

    # Create a batch for a specific subreddit and submit it.
    # Update the index so we know the batch id of a given subreddit.
    create_batch_for_subreddit(client, subreddit)

    # Check the status of your batch
    while True:
        status = check_batch_status(client, subreddit)

        if status["status"] == "completed":
            break

        print(
            f"Status: {status['status']}\n"
            f"Completed: {status['completed']}\n"
            f"Failed: {status['failed']}\n"
            f"Total: {status['total']}   "
        )
        time.sleep(1)
        print()

    # Write result of batch to jsonl file inside batch_results
    # Update the index so we know the file id of the batch result
    create_batch_result(client, subreddit)


def main():
    load_dotenv()

    # NOTE: This should only be done once
    # Submit the preprocessed data to OpenAI servers.
    # Create an index to know the file id of each subreddit.
    # submit_all_subreddits()

    # Pick your subreddit
    subreddit = "books"

    start_subreddit_batch_pipeline(subreddit)

    # Check stats for a given subreddit
    print_stats_for_subreddit(subreddit)


if __name__ == "__main__":
    main()
