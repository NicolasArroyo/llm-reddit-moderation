import os
import csv
import sys
import time
import json
import datetime
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from markdown_utils import unmark
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ONLY_SOME_SUBREDDITS = True
ONLY_SOME_SUBREDDITS_LIST = ['worldnews', 'askscience']
SMALL_SUBSET_OF_COMMENTS_PER_SUBREDDIT = True
SMALL_SUBSET_OF_COMMENTS_PER_SUBREDDIT_NUMBER = 100

load_dotenv()


class Prediction(BaseModel):
    would_moderate: bool
    rule_nums: list[int]
    explanation: str
    rating: int


def preprocess_rules(rules):
    preprocessed_rules = ""

    for i, rule in enumerate(rules):
        if rule['kind'] == 'link':
            continue

        preprocessed_rules += f"{str(i + 1)}. {unmark(rule['description']).strip()}\n"

    return preprocessed_rules


def preprocess_comments(subreddit):
    csv_file_name = f'./data/rule_moderation/subreddit_balanced_datasets/{subreddit}.csv'

    comments = []
    with open(csv_file_name) as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            if SMALL_SUBSET_OF_COMMENTS_PER_SUBREDDIT and len(
                    comments) == SMALL_SUBSET_OF_COMMENTS_PER_SUBREDDIT_NUMBER:
                break

            comments.append({
                'comment': row[0],
                'label': True if row[2] == '1' else False
            })

    return comments


def preprocess_data():
    data_dict = {}

    with open('./data/rule_moderation/subreddit_rules_w_description.jsonl') as file:
        for line in file:
            line_json = json.loads(line)

            if ONLY_SOME_SUBREDDITS and line_json['subreddit'] not in ONLY_SOME_SUBREDDITS_LIST:
                continue

            data_dict[line_json['subreddit']] = {
                'description': line_json['description'],
                'rules': preprocess_rules(line_json['rules']),
                'comments': preprocess_comments(line_json['subreddit'])
            }

    return data_dict


def make_user_dict(string):
    return {
        "role": "user",
        "content": string
    }


def make_assistant_dict(string):
    return {
        "role": "assistant",
        "content": string
    }


def make_chat(subreddit, description, rules, comment):
    base_chat = [
        {
            'role': 'system',
            'content': f'You are a helpful content moderation assistant for Reddit. In this opportunity your main job will be moderating an online subreddit called {subreddit}. This subreddit has the following description: {description}. Here are the rules for the community:{rules}. Consider the following comment: {comment}'
        },
    ]

    return base_chat


def print_statistics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(f"True labels: {len(true_labels)}")
    print(f"Predicted labels: {len(predicted_labels)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1-Score: {f1:.2%}")
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


def get_statistics_dict(true_labels, predicted_labels):
    accuracy = round(accuracy_score(true_labels, predicted_labels), 2)
    precision = round(precision_score(true_labels, predicted_labels), 2)
    recall = round(recall_score(true_labels, predicted_labels), 2)
    f1 = round(
        f1_score(true_labels, predicted_labels), 2)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def save_file_json(path, prefix, data):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_file = f'{path}/{prefix}_{timestamp}.json'

    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

    print(f'FILE SAVED CORRECTLY TO {output_file}')


def main():
    client = OpenAI(
        api_key=os.getenv("OPEN_AI_KEY")
    )

    true_labels = []
    predicted_labels = []

    data = preprocess_data()

    for subreddit in data:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(subreddit)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        description = data[subreddit]['description']
        rules = data[subreddit]['rules']

        for count, comment_dict in enumerate(data[subreddit]['comments']):
            print(f'PROCESSING COMMENT {count}')
            comment = comment_dict['comment']
            mod_chat = make_chat(subreddit, description, rules, comment)

            try:
                completion = client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=mod_chat,
                    temperature=0,
                    response_format=Prediction
                )

                prediction = completion.choices[0].message

                if prediction.parsed:
                    comment_dict['prediction'] = json.loads(prediction.content)
                    true_labels.append(comment_dict['label'])
                    predicted_labels.append(comment_dict['prediction']['would_moderate'])

                elif prediction.refusal:
                    print(f'COMMENT {count} REFUSAL')
                    print(prediction.refusal)


            except Exception as e:
                print(f'COMMENT {count} EXCEPTION')
                print(e)
                pass

        data[subreddit]['statistics'] = get_statistics_dict(true_labels, predicted_labels)

    save_file_json('./runs', 'run', data)


if __name__ == '__main__':
    main()
