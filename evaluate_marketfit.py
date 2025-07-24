import os
import json
import re
from groq import Groq
from dotenv import load_dotenv


# rubrics.py

X_AXIS_CATEGORIES = [
    "not relevant to the Idea ",
    "Target Audience Alignment only",
    "Problem-Solution Alignment only",
    "Customer Validation Evidence only",
    "Unique Value Proposition only",
    "Outlines initial steps for entering the market only",
    "Target Audience Alignment + Problem-Solution Alignment",
    "Target Audience Alignment + Customer Validation Evidence",
    "Target Audience Alignment + Unique Value Proposition",
    "Target Audience Alignment + Outlines initial steps",
    "Problem-Solution Alignment + Customer Validation Evidence",
    "Problem-Solution Alignment + Unique Value Proposition",
    "Problem-Solution Alignment + Outlines initial steps",
    "Customer Validation Evidence + Unique Value Proposition",
    "Customer Validation Evidence + Outlines initial steps",
    "Unique Value Proposition + Outlines initial steps",
    "Target Audience Alignment + Problem-Solution Alignment + Customer Validation Evidence",
    "Target Audience Alignment + Problem-Solution Alignment + Unique Value Proposition",
    "Target Audience Alignment + Problem-Solution Alignment + Outlines initial steps",
    "Target Audience Alignment + Customer Validation Evidence + Unique Value Proposition",
    "Target Audience Alignment + Customer Validation Evidence + Outlines initial steps",
    "Target Audience Alignment + Unique Value Proposition + Outlines initial steps",
    "Problem-Solution Alignment + Customer Validation Evidence + Unique Value Proposition",
    "Problem-Solution Alignment + Customer Validation Evidence + Outlines initial steps",
    "Problem-Solution Alignment + Unique Value Proposition + Outlines initial steps",
    "Customer Validation Evidence + Unique Value Proposition + Outlines initial steps",
    "Target Audience Alignment + Problem-Solution Alignment + Customer Validation Evidence + Unique Value Proposition",
    "Target Audience Alignment + Problem-Solution Alignment + Customer Validation Evidence + Outlines initial steps",
    "Target Audience Alignment + Problem-Solution Alignment + Unique Value Proposition + Outlines initial steps",
    "Target Audience Alignment + Customer Validation Evidence + Unique Value Proposition + Outlines initial steps",
    "Problem-Solution Alignment + Customer Validation Evidence + Unique Value Proposition + Outlines initial steps",
    "Target Audience Alignment + Problem-Solution Alignment + Customer Validation Evidence + Unique Value Proposition + Outlines initial steps"
]

Y_AXIS_CATEGORIES = [
    "not relevant to the Idea ",
    "Does not have grammar",
    "Does not Demonstrate Understanding Only",
    "Is not Precise and To the Point",
    "Info is not Well-Structured and Is not Easy to Understand",
    "Does not have Grammar + Does not Demonstrate Understanding",
    "Does not have Grammar + Is not Precise and To the Point",
    "Does not have Grammar + Info is not Well-Structured and Is not Easy to Understand",
    "Does not Demonstrate Understanding + Info is not Well-Structured and Is not Easy to Understand",
    "Does not Demonstrate Understanding + Is not Precise and To the Point",
    "Is not Precise and To the Point + Info is not Well-Structured and Is not Easy to Understand",
    "Does not have Grammar + Does not Demonstrate Understanding + Is not Precise and To the Point",
    "Does not have Grammar + Does not Demonstrate Understanding + Info is not Well-Structured and Is not Easy to Understand",
    "Does not have Grammar + Is not Precise and To the Point + Info is not Well-Structured and Is not Easy to Understand",
    "Does not Demonstrate Understanding + Is not Precise and To the Point + Info is not Well-Structured and Is not Easy to Understand",
    "Does not have Grammar + Does not Demonstrate Understanding + Is not Precise and To the Point + Info is not Well-Structured and Is not Easy to Understand",
    "Has some grammar",
    "Demonstrates some Understanding",
    "Is somewhat Precise and To the Point",
    "Info is somewhat Well-Structured and fairly Easy to Understand",
    "Has some Grammar + Demonstrates some Understanding",
    "Has some Grammar + is somewhat Precise and To the Point",
    "Has some Grammar + Info is somewhat Well-Structured and fairly Easy to Understand",
    "Demonstrates some Understanding + Info is somewhat Well-Structured and fairly Easy to Understand",
    "Demonstrates some Understanding + is somewhat Precise and To the Point",
    "Is somewhat Precise and To the Point + Info is somewhat Well-Structured and is somewhat Easy to Understand",
    "Has some Grammar + Demonstrates some Understanding + is somewhat Precise and To the Point",
    "Has some Grammar + Demonstrates some Understanding + Info is somewhat Well-Structured and somewhat Easy to Understand",
    "Has some Grammar + is somewhat Precise and To the Point + Info is somewhat Well-Structured and is somewhat Easy to Understand",
    "Demonstrates some Understanding + is somewhat Precise and To the Point + Info is somewhat Well-Structured and is somewhat Easy to Understand",
    "Has some Grammar + Demonstrates some Understanding + is somewhat Precise and To the Point + Info is somewhat Well-Structured and somewhat Easy to Understand",
    "Has Very good Grammar + Demonstrates Very good Understanding",
    "Has Very good Grammar + Is Precise and To the Point",
    "Has Very good Grammar + Info is Well-Structured and Easy to Understand",
    "Demonstrates Very good Understanding + Is Precise and To the Point",
    "Demonstrates Very Good Understanding + Info is Very Well-Structured and Easy to Understand",
    "Is Precise and To the Point + Info is Very Well-Structured and Easy to Understand",
    "Has very good Grammar + Demonstrates Very Good Understanding + Is Precise and To the Point",
    "Has Very Good Grammar + Demonstrates Very Good Understanding + Info is Very Well-Structured and Easy to Understand",
    "Has Very Good Grammar + Is Precise and To the Point + Info is Very Well-Structured and Easy to Understand",
    "Demonstrates Very Good Understanding + Is Precise and To the Point + Info is Very Well-Structured and Easy to Understand",
    "Has Very Good Grammar + Demonstrates Very Good Understanding + Is Precise and To the Point + Info is Very Well-Structured and Easy to Understand"
]


load_dotenv()  # Load variables from .env

PROMPT_TEMPLATE = '''You are a specialized Assessment Agent designed to evaluate student-written "market fit" data submissions from students aged 14-16 years, focusing on United Nations Sustainable Development Goals (SDGs).

## Output Requirement
Return ONLY this exact JSON structure:
{{"X_Axis_Rubric_Category": "<category>", "Y_Axis_Rubric_Category": "<category>"}}

## X-Axis Categories:
{X_AXIS_LIST}

## Y-Axis Categories:
{Y_AXIS_LIST}

Evaluate this submission:
{submission_text}

Return ONLY the JSON.

dont mention the index of the categories, just return the category name.

'''

def evaluate_market_fit_with_groq(submission_text: str) -> dict:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"error": "Missing GROQ_API_KEY in .env"}

    client = Groq(api_key=api_key)

    x_axis_list = "\n".join([f"{i+1}. {item}" for i, item in enumerate(X_AXIS_CATEGORIES)])
    y_axis_list = "\n".join([f"{i+1}. {item}" for i, item in enumerate(Y_AXIS_CATEGORIES)])

    prompt = PROMPT_TEMPLATE.format(
        X_AXIS_LIST=x_axis_list,
        Y_AXIS_LIST=y_axis_list,
        submission_text=submission_text
    )

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a specialized assessment agent. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )

        raw = response.choices[0].message.content.strip()

        # Try direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"error": "Invalid JSON format", "raw": raw}

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    try:
        print("Starting market fit evaluation...", flush=True)
        submission = "my product is useful in rural areas for making them solar powered and internet free. It is a solar powered internet device that can be used in rural areas to provide internet access and solar power. It is designed to be affordable and easy to use, making it accessible to people in rural areas who may not have access to traditional internet services or electricity."
        result = evaluate_market_fit_with_groq(submission)
        print("Result:", flush=True)
        print(result, flush=True)
    except Exception as e:
        print(f"Exception in main: {e}", flush=True)