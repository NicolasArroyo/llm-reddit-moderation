import os
import csv
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
SMALL_SUBSET_OF_COMMENTS_PER_SUBREDDIT_NUMBER = 20

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
            }

    return data_dict


def make_chat(subreddit, description, rules, comment):
    base_chat = [
        {
            'role': 'system',
            'content': f'You are a helpful content moderation assistant for Reddit. In this opportunity your main job will be moderating an online subreddit called {subreddit}. This subreddit has the following description: {description}. Here are the rules for the community:{rules}. Consider the following comment: {comment}'
        },
    ]

    return base_chat


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


def main():
    client = OpenAI(
        api_key=os.getenv("OPEN_AI_KEY")
    )

    data = preprocess_data()

    timestamp = datetime.datetime.now().strftime(f'%Y-%m-%d_%H-%M-%S_{SMALL_SUBSET_OF_COMMENTS_PER_SUBREDDIT_NUMBER}')
    os.makedirs(f'./results/{timestamp}')

    for subreddit in data:
        true_labels = []
        predicted_labels = []

        os.makedirs(f'./results/{timestamp}/{subreddit}')

        description = data[subreddit]['description']
        rules = data[subreddit]['rules']

        subreddit_result = {
            'description': description,
            'rules': rules,
            'comments': []
        }

        for count, comment_dict in enumerate(preprocess_comments(subreddit)):
            comment = comment_dict['comment']
            label = comment_dict['label']

            mod_chat = make_chat(subreddit, description, rules, comment)

            print(f'{subreddit}: Comment {count}')

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
                    true_labels.append(label)
                    predicted_labels.append(comment_dict['prediction']['would_moderate'])
                    subreddit_result['comments'].append(comment_dict)

                elif prediction.refusal:
                    print(f'{subreddit}: Comment {count} refusal is True')
                    print(prediction.refusal)


            except Exception as e:
                print(f'{subreddit}: Comment {count} gave exception: {e}')
                pass

            subreddit_result['statistics'] = get_statistics_dict(true_labels, predicted_labels)

        with open(f'./results/{timestamp}/{subreddit}/results.json', 'w', encoding='utf-8') as file:
            json.dump(subreddit_result, file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
