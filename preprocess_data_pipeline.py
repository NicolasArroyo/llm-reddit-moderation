import os
import csv
import json

from markdown_utils import unmark


# Removes Markdown format from the rules of a subreddit
def preprocess_rules(rules):
    preprocessed_rules = ""

    for i, rule in enumerate(rules):
        if rule["kind"] == "link":
            continue

        # Rules will be returned as:
        # 1. Rule 1\n
        # 2. Rule 2\n
        # ...
        preprocessed_rules += f"{str(i + 1)}. {unmark(rule['description']).strip()}\n"

    return preprocessed_rules


# Creates a list of dicts with the comment and its label
def preprocess_comments(subreddit):
    csv_file_name = (
        f"./data/rule_moderation/subreddit_balanced_datasets/{subreddit}.csv"
    )

    comments = []
    with open(csv_file_name) as file:
        reader = csv.reader(file)
        next(reader)  # Skips the header

        for row in reader:
            comments.append(
                {"comment": row[0], "label": True if row[2] == "1" else False}
            )

    return comments


# Returns initial chat with GPT with the context of the interaction
def make_chat(subreddit, description, rules, comment):
    base_chat = [
        {
            "role": "system",
            "content": f"You are a helpful content moderation assistant for the online subreddit called {subreddit}.",
        },
        {
            "role": "system",
            "content": f"The subreddit has the following description: {description}.",
        },
        {
            "role": "system",
            "content": f"Here are the rules for the subreddit: {rules}",
        },
        {
            "role": "system",
            "content": f"Here is a description of the parameters of the response schema provided:\n- would_moderate: boolean if you would or not moderate this comment.\n- rule_nums: a comma-separated list of rules being violated.\n- explanation: string with the reason for your decision.\n- rating: a score from 1-5 on how violative the comment is",
        },
        {
            "role": "system",
            "content": f"Consider the following comment: {comment}",
        },
    ]

    return base_chat


def main():
    preprocessed_data_path = "./preprocessed_data"
    os.makedirs(preprocessed_data_path)  # Where we'll store the preprocessed data

    with open(
        f"./data/rule_moderation/subreddit_rules_w_description.jsonl"
    ) as srd_file:
        for line in srd_file:
            line_dict = json.loads(line)

            subreddit = line_dict["subreddit"]
            description = line_dict["description"]
            rules = preprocess_rules(line_dict["rules"])
            comment_dicts = preprocess_comments(subreddit)

            # We'll create a new jsonl file for each subreddit
            # Example: ./preprocessed_data/worldnews.jsonl
            with open(f"./preprocessed_data/{subreddit}.jsonl", "a") as subreddit_file:
                for count, comment_dict in enumerate(comment_dicts):
                    comment = comment_dict["comment"]

                    chat = make_chat(subreddit, description, rules, comment)

                    jsonl_line = {
                        "custom_id": f"comment_{count}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": "gpt-4o-mini",
                            "messages": chat,
                            "response_format": {
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "prediction_response",
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "would_moderate": {"type": "boolean"},
                                            "rule_nums": {
                                                "type": "array",
                                                "items": {"type": "integer"},
                                            },
                                            "explanation": {"type": "string"},
                                            "rating": {"type": "integer"},
                                        },
                                        "required": [
                                            "would_moderate",
                                            "rule_nums",
                                            "explanation",
                                            "rating",
                                        ],
                                        "additionalProperties": False,
                                    },
                                    "strict": True,
                                },
                            },
                            "temperature": 0,
                        },
                    }

                    subreddit_file.write(json.dumps(jsonl_line) + "\n")


if __name__ == "__main__":
    main()
