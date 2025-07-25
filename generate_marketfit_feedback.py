from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_market_fit_feedback(marketfit, openai_api_key=None):
    """
    Evaluates a student's market fit response using OpenAI API.

    Args:
        marketfit (str): Student's response about their business idea's market fit
        openai_api_key (str, optional): Your OpenAI API key. If not provided, will load from OPENAI_API_KEY environment variable
    
    Returns:
        str: Detailed feedback based on the 10-point rubric
    """
    
    # Get API key from parameter or environment variable
    if openai_api_key is None:
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            return "Error: OPENAI_API_KEY not found. Please set it in your environment variables or pass it as a parameter."
    
    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)
    
    rubric_prompt = """
You are an expert evaluator. A student has written a response to the prompt:

"Write why you believe your idea is needed in the market and how your idea is unique. Use any data or current knowledge you have. Outline how you will enter the market."

Evaluate the student's response using the following 10-point rubric. For each point, give specific feedback. If a point is missing or weak, suggest how to improve it.

you are speaking to the student , so never use "the student's response".
1. Target Audience Alignment – Does it clearly define the target audience and their needs?
2. Problem-Solution Alignment – Does it explain how the idea addresses the customer needs?
3. Customer Validation Evidence – Is there any supporting evidence like surveys, interviews, pilot results?
4. Unique Value Proposition – What makes the solution unique or better than others?
5. Initial Market Entry Plan – Are there steps for MVP, pilots, or customer onboarding?
6. Grammar – Is the grammar, spelling, and punctuation correct?
7. Demonstrates Understanding – Does the student show clear insight and comprehension of the topic?
8. Precise and To the Point – Is the answer concise and avoids unnecessary content?
9. Relevant to the Idea – Does the content stay focused on the idea and its value?
10. Info is Well-Structured and Easy to Understand – Is the writing organized and clear?


Return the feedback in a numbered format.
"""
    
    # Check if input is empty or just whitespace
    if not marketfit or not marketfit.strip():
        return "I'd love to help evaluate your business idea! Please share your thoughts about: What business idea do you have? What problem does it solve? Who would be your customers? What makes your idea unique? How would you get started?"
    
    # Construct the full prompt
    full_prompt = rubric_prompt + "\n\nStudent Response:\n" + marketfit.strip()
    
    try:
        # Make API call to OpenAI
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": full_prompt,
                }
            ],
            model="gpt-4.1",
            temperature=0.7,
            max_tokens=2048,
            top_p=0.9,
            presence_penalty=0.1,
            frequency_penalty=0.1
        )
        
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        return f"Error generating feedback: {str(e)}"


# Example usage:
if __name__ == "__main__":  
    # API key will be loaded from environment variable OPENAI_API_KEY
    # Make sure to create a .env file with: OPENAI_API_KEY=your-actual-api-key
    
    # Example student response
    student_response = """
    My business idea is a mobile app that helps students find study groups. 
    Students often struggle to find people to study with, especially in large classes. 
    My app would connect students based on their courses, study preferences, and location. 
    I think this is unique because current solutions don't focus specifically on academic collaboration.
    I would start by testing it at my school first, then expand to other universities.
    """
    
    # Get feedback (API key loaded from environment)
    feedback = get_market_fit_feedback(student_response)
    print(feedback)