#!/usr/bin/env python
import sys
import warnings
from datetime import datetime

from news_analysis.crew import NewsAnalysis

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information
# define the current year and month
current_year_month = datetime.now().strftime("%Y-%m")

def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': 'AI Safety',
        'current_year_month': current_year_month
    }
    NewsAnalysis().crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI Safety"
    }
    try:
        NewsAnalysis().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        NewsAnalysis().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        NewsAnalysis().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")
