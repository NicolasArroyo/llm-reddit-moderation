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


def submit_all_subreddits(client):
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

    print(f"Status: {status}\n{completed} completed | {failed} failed\nTotal = {total}")


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


def main():
    load_dotenv()

    client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

    # First step:
    # 1. Submit the preprocessed data to OpenAI servers.
    # 2. Create an index to know the file id of each subreddit.
    # submit_all_subreddits(client)

    # Second step:
    # 1. Pick your subreddit
    # specific_subreddit = "worldnews"

    # Third step:
    # 1. Create a batch for a specific subreddit and submitting it.
    # 2. Update the index so we know the batch id of a given subreddit.
    # create_batch_for_subreddit(client, specific_subreddit)

    # Fourth step:
    # 1. Check the status of your batch
    # 2. Repeat step 1
    # while True:
    #     check_batch_status(client, specific_subreddit)
    #     time.sleep(1)
    #     print()

    # Fifth step:
    # 1. Write result of batch to jsonl file inside batch_results
    # 2. Update the index so we know the file id of the batch result
    # create_batch_result(client, specific_subreddit)

    # Sixth step
    # 1. Check stats for a given subreddit
    # print_stats_for_subreddit(specific_subreddit)


if __name__ == "__main__":
    main()
