<br>
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="images/logo.png" alt="Logo" width="150" height="150">
  </a>
  <h3 align="center">Reddit Content Moderation with LLMs</h3>
  <p align="center">
    Exploring the application of Large Language Models for efficient content moderation on Reddit
  </p>
</div>

---

## About the Project

**Reddit Content Moderation with LLMs** is a research-driven project based on the paper [*"Watch Your Language:
Investigating Content Moderation with Large Language
Models"*](https://ojs.aaai.org/index.php/ICWSM/article/view/31358/33518). This project focuses on the first phase of the
study, which evaluates the rule-based moderation capabilities of Large Language Models (LLMs).

The primary objective of the project is to optimize the content moderation pipeline by significantly reducing
operational costs through two key improvements:

- **GPT-4o Mini**: Leveraging OpenAI's cost-efficient model instead of GPT-3.5 Turbo. This model dramatically lowers the
  cost of processing a million input tokens from \$3.00 to \$0.15 and output tokens from \$6.00 to \$0.60.
- **Batch API**: Utilizing OpenAI's Batch API to further reduce token processing costs by 50%. This approach requires
  batch submission of requests, with responses provided within 24 hours.

---

## Getting Started

### Prerequisites

To use this project, ensure you have the following:

- **OpenAI API Key**: Required for making API calls.
- **OpenAI Account with Credit Balance (Optional)**: Enables higher API usage within shorter timeframes.

## Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/NicolasArroyo/llm-reddit-moderation.git
   ```

2. **Navigate to the project directory**:
   ```sh
   cd llm-reddit-moderation
   ```

3. **Install the required dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

4. **Create a `.env` file with the following key**:
   ```
   OPEN_AI_KEY="YOUR API KEY"
   ```

5. **Download the curated data** from the [official repository](https://github.com/kumarde/llm-content-mod) of the
   research paper and place it in the project folder.

6. **Ensure your project directory is structured as follows**:
   ```
   llm-reddit-moderation/
   ├── batch_pipeline
   ├── markdown_utils.py
   ├── preprocess_data_pipeline.py
   ├── requirements.txt
   └── data/
       ├── subreddit_rules_w_description.jsonl
       └── subreddit_balanced_datasets/
   ```

---

## Usage

To run the project for the first time do the following:

1. **Run the preprocessing script** to create `.jsonl` files for each subreddit that will be sent through the Batch API:
   ```sh
   python3 preprocess_data_pipeline.py
   ```

2. **Ensure the function call to `submit_all_subreddits` in `batch_pipeline.py` is uncommented.** This step is required
   to upload the preprocessed data to OpenAI servers before performing the content moderation task for the first time.

3. **Set the subreddit** in `batch_pipeline.py` by modifying the `subreddit` variable in the main function to the
   subreddit you want to analyze:
   ```python
   subreddit = "askscience"
   ```

4. **Execute `batch_pipeline.py`** to begin processing. Monitor the terminal for real-time updates on the status of your
   batch requests as they are processed by OpenAI:
   ```sh
   python3 batch_pipeline.py
   ```

5. **View the results**: Once processing is complete, the terminal will display metrics summarizing the content
   moderation analysis for the selected subreddit:
   ```
   {'accuracy': 0.75, 'precision': 0.682, 'recall': 0.936, 'f1': 0.789}
   ```

All Batch API results will be stored in the `batch_results` directory, organized by subreddit:

```
llm-reddit-moderation/
    └── batch_results/
        ├── askscience.jsonl
        └── worldnews.json
```

Additionally, an index file named `subreddit_batch_file_index.jsonl` will be maintained to track the following details:

- **file_id**: The unique identifier for the preprocessed data file associated with each subreddit.
- **batch_id**: The unique identifier for each batch submission via `batch_pipeline.py` for a specific subreddit.
- **response_file_id**: The unique identifier for the response file generated for each subreddit batch submission.

---

## Findings

Our analysis builds on the cost and time comparisons derived from Table 10 of the referenced research paper:

| **Model** | **Cost ($)** | **Time Taken (days)** |
|-----------|--------------|-----------------------|
| GPT-3.5   | 175          | 5                     |

### Scope of Analysis

Due to the financial constraints of running the entire pipeline across all 95 subreddits, we selected a subset of
subreddits for analysis. These include:

- **r/askscience**
- **r/IAmA**
- **r/movies**
- **r/worldnews**
- **r/depression** *(included due to its significant size, despite not being mentioned in the paper)*

---

### Cost Analysis

The pipeline was executed for the selected subreddits, producing the following cost breakdown:

| **Subreddit**  | **Input Tokens Cost ($)** | **Output Tokens Cost ($)** | **Total Cost ($)** |
|----------------|---------------------------|----------------------------|--------------------|
| **askscience** | 0.0351                    | 0.0177                     | 0.0529             |
| **IAmA**       | 0.0251                    | 0.0158                     | 0.0409             |
| **movies**     | 0.0648                    | 0.0177                     | 0.0825             |
| **worldnews**  | 0.0283                    | 0.0183                     | 0.0466             |
| **depression** | 0.0864                    | 0.0237                     | 0.1102             |

#### Cost Projection for All Subreddits

Using **r/depression** as the worst-case scenario (highest cost), we estimate the total cost to process all 95
subreddits at approximately **\$10.47**, a substantial reduction from the original **\$175** reported for GPT-3.5.

**Note:** This is a conservative estimate. A full pipeline run across all subreddits is necessary to obtain precise cost
figures.

---

### Time Analysis

According to OpenAI's Batch API documentation, all requests are guaranteed to complete within a maximum time frame of
**24 hours**. This represents a significant improvement over the previously reported 5-day processing time for GPT-3.5.

In practice, batch completion times might be faster than this worst-case scenario. Our experimental results for
the selected subreddits are summarized below:

| **Subreddit**  | **Time Taken (minutes)** |
|----------------|--------------------------|
| **askscience** | 21                       |
| **IAmA**       | 16                       |
| **movies**     | 20                       |
| **worldnews**  | 39                       |
| **depression** | 17                       |

---

## Acknowledgments

This project utilizes curated data, a modified prompt, and an adapted response format derived from the research paper [
*Watch Your Language: Investigating Content Moderation with Large Language
Models*](https://ojs.aaai.org/index.php/ICWSM/article/view/31358/33518), specifically
its [official repository](https://github.com/kumarde/llm-content-mod).

Special thanks to Deepak Kumar, Yousef Anees AbuHashem, and Zakir Durumeric for their work, which inspired further
exploration into cost-effective approaches to content moderation.
