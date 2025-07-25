import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
import logging
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

def load_api_keys_from_env() -> Dict[str, str]:
    """
    Load API key from environment variables.
    
    Returns:
        Dict with OpenAI API key
    """
    try:
        keys = {
            'openai': os.getenv("OPENAI_API_KEY")
        }
        
        available_keys = {k: v for k, v in keys.items() if v}
        logger.info(f"Available API keys: {list(available_keys.keys())}")
        
        return keys
    except Exception as e:
        logger.error(f"Error loading API keys from environment: {e}")
        return {'openai': None}

def create_validation_prompt(idea: str, problem: str, prototype_description: str) -> str:
    """
    Create a focused prompt template for content validation.
    
    Args:
        idea: Student's innovation idea
        problem: Problem statement
        prototype_description: Prototype description
    
    Returns:
        Formatted prompt for Groq API
    """
    
    prompt = f"""You are an educational content validator for a student innovation platform. Your job is to evaluate if student submissions are appropriate, educational, and coherent.

VALIDATION CRITERIA:
1. SAFETY (STRICT): Content must be appropriate for educational use and MUST NOT include:
   - Any reference to weapons, security systems, or surveillance
   - Violence or potential for harm
   - Adult content or inappropriate themes
   - Illegal activities or substances
   - Hate speech or discriminatory content
   - Content that may cause anxiety or distress
2. COHERENCE: The idea, problem, and prototype must be logically related and form a coherent solution
3. EDUCATIONAL VALUE: Content should demonstrate learning, innovation, problem-solving, or positive impact
   - Focus on academic, social, or environmental improvement
   - Emphasis on constructive solutions
   - Avoid controversial or sensitive topics

STUDENT SUBMISSION:
---
IDEA: {idea}

PROBLEM: {problem}

PROTOTYPE DESCRIPTION: {prototype_description}
---

EVALUATION INSTRUCTIONS:
- Analyze the submission against all three criteria
- Check if the idea logically addresses the stated problem
- Verify the prototype description implements the proposed idea
- Ensure all content is safe and appropriate for students
- Look for educational merit and innovation focus

RESPONSE FORMAT (JSON only):
{{
    "valid": true/false,
    "message": "Brief explanation of decision (max 150 characters)"
}}

EXAMPLES:
Valid Examples:
- Idea: Smart recycling bin with AI sorting
  Problem: Students don't know how to sort waste
  Solution: IoT bin with camera recognition for waste classification
  (✓ Safe, educational, focused on environmental learning)

- Idea: Interactive plant growing system
  Problem: Students lack understanding of biology
  Solution: Smart greenhouse with sensors and growth tracking
  (✓ Safe, educational, promotes scientific learning)

Invalid Examples:
- Idea: Student tracking system
  Problem: Monitor student attendance
  Solution: Surveillance cameras with face recognition
  (✗ Privacy concerns, surveillance-related, may cause anxiety)

- Idea: Dating app for students
  Problem: Students need better social connections
  Solution: Social media platform with messaging
  (✗ Inappropriate for educational setting, safety concerns)

Remember: Any content involving surveillance, monitoring, or potential privacy violations should be marked as invalid.

Evaluate the submission now:"""

    return prompt

def validate_student_input(
    idea: str,
    problem: str,
    prototype_description: str,
    model: str = "gpt-4.1"  # Using OpenAI's GPT-4.1 model
) -> Dict[str, Any]:
    """
    Validate student input using Groq API for content safety and coherence.
    
    Args:
        idea (str): The innovative idea or solution concept
        problem (str): The specific problem statement being solved
        prototype_description (str): Detailed prototype description
        model (str): Groq model to use for validation
    
    Returns:
        Dict containing:
        - valid (bool): Whether the input passes validation
        - message (str): Explanation of the decision
        - error (str): Error message if API call fails
        - api_available (bool): Whether required APIs are available
    """
    
    # Load API keys
    api_keys = load_api_keys_from_env()
    
    if not api_keys['openai']:
        return {
            "valid": False,
            "message": "OpenAI API key not found in environment variables",
            "error": "Missing OPENAI_API_KEY",
            "api_available": False
        }
    
    # Basic input validation
    if not all([idea, problem, prototype_description]):
        return {
            "valid": False,
            "message": "All fields (idea, problem, prototype description) are required",
            "error": "Missing required fields",
            "api_available": True
        }
    
    # Length checks
    if len(idea.strip()) < 10:
        return {
            "valid": False,
            "message": "Idea must be at least 10 characters long",
            "error": "Idea too short",
            "api_available": True
        }
    
    if len(problem.strip()) < 20:
        return {
            "valid": False,
            "message": "Problem statement must be at least 20 characters long",
            "error": "Problem statement too short",
            "api_available": True
        }
    
    if len(prototype_description.strip()) < 50:
        return {
            "valid": False,
            "message": "Prototype description must be at least 50 characters long",
            "error": "Prototype description too short",
            "api_available": True
        }
    
    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=api_keys['openai'])
        logger.info("OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return {
            "valid": False,
            "message": "Failed to connect to validation service",
            "error": f"OpenAI client initialization failed: {str(e)}",
            "api_available": False
        }
    
    # Create validation prompt
    validation_prompt = create_validation_prompt(idea, problem, prototype_description)
    
    # Call OpenAI API for validation
    try:
        logger.info(f"Sending validation request to OpenAI using model: {model}")
        
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an educational content validator. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": validation_prompt
                }
            ],
            model=model,
            temperature=0.1,  # Low temperature for consistent validation
            max_tokens=200,   # Short response needed
            response_format={"type": "json_object"},
            presence_penalty=0.1,
            frequency_penalty=0.1
        )
        
        # Extract response content
        response_content = response.choices[0].message.content.strip()
        logger.info(f"Received response from OpenAI: {response_content[:100]}...")
        
        # Parse JSON response
        try:
            validation_result = json.loads(response_content)
            
            # Ensure required fields exist
            if 'valid' not in validation_result or 'message' not in validation_result:
                raise ValueError("Invalid response format from validation service")
            
            # Clean up message (ensure it's reasonable length)
            message = validation_result['message'][:200]  # Limit message length
            
            return {
                "valid": bool(validation_result['valid']),
                "message": message,
                "error": None,
                "api_available": True,
                "model_used": model,
                "openai_available": api_keys['openai'] is not None
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {response_content}")
            
            # Fallback: try to extract valid/invalid from text
            response_lower = response_content.lower()
            if 'valid": true' in response_lower or '"valid":true' in response_lower:
                return {
                    "valid": True,
                    "message": "Content appears appropriate for educational use",
                    "error": "JSON parsing failed but content seems valid",
                    "api_available": True
                }
            else:
                return {
                    "valid": False,
                    "message": "Content validation failed - please review your submission",
                    "error": "JSON parsing failed and content appears invalid",
                    "api_available": True
                }
    
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        
        # Handle specific error types
        error_message = str(e).lower()
        if "rate limit" in error_message:
            return {
                "valid": False,
                "message": "Validation service is currently busy. Please try again in a moment.",
                "error": "Rate limit exceeded",
                "api_available": True
            }
        elif "api key" in error_message or "unauthorized" in error_message:
            return {
                "valid": False,
                "message": "Validation service configuration error",
                "error": "Invalid API key",
                "api_available": False
            }
        else:
            return {
                "valid": False,
                "message": "Validation service is temporarily unavailable",
                "error": f"API error: {str(e)}",
                "api_available": True
            }

# Helper function to validate with default model
def validate_with_model(idea: str, problem: str, prototype_description: str, model_name: str = None) -> Dict[str, Any]:
    """
    Validate using OpenAI GPT-4.1 model.
    """
    return validate_student_input(idea, problem, prototype_description, "gpt-4.1")

# Example usage and testing
if __name__ == "__main__":
    print("=== Testing Student Input Validation with OpenAI GPT-4.1 ===\n")
    
    # Test 1: Good educational content
    print("Test 1: Valid Educational Content")
    result1 = validate_student_input(
        idea="Smart water bottle that reminds students to stay hydrated during study sessions",
        problem="Students often forget to drink water while studying, leading to dehydration and reduced focus",
        prototype_description="IoT-enabled water bottle with LED indicators, smartphone app integration, customizable reminder intervals, and hydration tracking. Features include study mode integration, daily goals, and health insights."
    )
    print(f"Valid: {result1['valid']}")
    print(f"Message: {result1['message']}")
    print(f"Error: {result1['error']}")
    print(f"Model used: {result1.get('model_used', 'unknown')}")
    print()
    
    # Test 2: Unrelated content
    print("Test 2: Unrelated Content")
    result2 = validate_student_input(
        idea="Gaming console for entertainment",
        problem="Students need better study habits",
        prototype_description="Social media platform for sharing photos and videos with friends"
    )
    print(f"Valid: {result2['valid']}")
    print(f"Message: {result2['message']}")
    print(f"Model used: {result2.get('model_used', 'unknown')}")
    print()
    
    # Test 3: Inappropriate content
    print("Test 3: Inappropriate Content")
    result3 = validate_student_input(
        idea="Student Monitoring System",
        problem="Teachers need to monitor student behavior",
        prototype_description="AI surveillance system with facial recognition to track student movements and behaviors, reporting any suspicious activities to authorities"
    )
    print(f"Valid: {result3['valid']}")
    print(f"Message: {result3['message']}")
    print(f"Model used: {result3.get('model_used', 'unknown')}")
    print()
    
 