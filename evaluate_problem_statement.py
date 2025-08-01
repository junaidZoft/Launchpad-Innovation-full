"""
Problem Statement Classifier - Optimized OpenAI Version

A high-performance classifier for evaluating student problem statements using OpenAI's GPT-4.1 API.
Optimized for speed with comprehensive error handling, caching, and batch processing.
"""

import os
import json
import time
import logging
import asyncio
import hashlib
from typing import Dict, Any, Optional, Tuple, List
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import aiohttp
from contextlib import asynccontextmanager

try:
    from openai import OpenAI, AsyncOpenAI
except ImportError as e:
    raise ImportError(
        "OpenAI library not found. "
        "Install with: pip install openai python-dotenv aiohttp"
    ) from e

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.warning("python-dotenv not found. Environment variables must be set manually.")


class ClassificationError(Exception):
    """Custom exception for classification errors."""
    pass


class APIError(ClassificationError):
    """API-related errors."""
    pass


class ConfigurationError(ClassificationError):
    """Configuration-related errors."""
    pass


class ValidationError(ClassificationError):
    """Input validation errors."""
    pass


@dataclass
class ClassificationResult:
    """Structured result from classification."""
    success: bool
    x_axis_category: Optional[str] = None
    y_axis_category: Optional[str] = None
    error: Optional[str] = None
    raw_response: Optional[str] = None
    processing_time: Optional[float] = None
    tokens_used: Optional[int] = None
    model_used: Optional[str] = None


class QualityLevel(Enum):
    """X-axis quality levels for validation."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    EXCEPTIONAL = "exceptional"


class ContentLevel(Enum):
    """Y-axis content levels for validation."""
    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    QUADRUPLE = "quadruple"
    QUINTUPLE = "quintuple"
    EXCEPTIONAL = "exceptional"


class OptimizedProblemStatementClassifier:
    """
    High-performance classifier using OpenAI's GPT-4.1 for fast and accurate inference.
    
    Features:
    - Ultra-fast GPT-4.1 integration
    - Advanced caching with TTL
    - Batch processing with connection pooling
    - Optimized prompt templates
    - Performance monitoring and metrics
    - Async/await support throughout
    """
    
    # Available OpenAI models (optimized for this use case)
    FAST_MODELS = [
        "gpt-4.1",    # Only available model
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4.1",  # Default to fastest model
        max_retries: int = 2,  # Reduced retries for speed
        timeout: float = 15.0,  # Reduced timeout
        enable_caching: bool = True,
        cache_size: int = 256,  # Increased cache size
        cache_ttl: int = 3600,  # Cache TTL in seconds
        log_level: str = "INFO",
        max_concurrent: int = 10  # For batch processing
    ):
        """Initialize the optimized classifier."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout = timeout
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.max_concurrent = max_concurrent
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        self._validate_configuration()
        
        # Initialize OpenAI clients
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        
        # Initialize cache with TTL
        self._cache = {}
        self._cache_timestamps = {}
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_response_time': 0.0,
            'total_tokens': 0,
            'fastest_response': float('inf'),
            'slowest_response': 0.0
        }
        
        # Thread pool for sync operations
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        
        self.logger.info(f"Groq Classifier initialized with model: {self.model_name}")
    
    def _validate_configuration(self) -> None:
        """Validate configuration parameters."""
        if not self.api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY not found. Set it as an environment variable or pass it directly."
            )
        
        if len(self.api_key.strip()) < 40:  # OpenAI keys are longer
            raise ConfigurationError("API key appears to be invalid (too short)")
        
        if self.model_name not in self.FAST_MODELS:
            self.logger.warning(f"Model {self.model_name} not in recommended fast models: {self.FAST_MODELS}")
        
        if self.max_retries < 1:
            raise ConfigurationError("max_retries must be at least 1")
        
        if self.timeout <= 0:
            raise ConfigurationError("timeout must be positive")
    
    @lru_cache(maxsize=1)
    def _get_optimized_prompt(self) -> str:
        """Get the optimized system prompt (cached)."""
        return '''
You are a specialized Assessment Agent designed to evaluate problem statements written by students aged 14-16 years, focusing on United Nations Sustainable Development Goals (SDGs). Your primary function is to provide consistent, objective analysis that remains stable across multiple evaluations of the same content.

## Core Mission
Analyze student-written problem statements with precision and consistency, ensuring that repeated assessments of identical content yield identical results. Your evaluation must be thorough, fair, and appropriate for the developmental level of teenage students. The provided IDEA will help you assess whether the problem statement is relevant and well-aligned with the intended solution concept.

## Assessment Framework
You must classify each problem statement using TWO dimensions:
- X-Axis (Quality Assessment): Evaluating the technical and structural quality of the writing, INCLUDING relevance to the provided idea
- Y-Axis (Content Assessment): Identifying what substantive elements are present in the problem statement

## Output Requirements
Your response MUST be a single, valid JSON object with this EXACT structure:

```json
{{
  "X_Axis_Rubric_Category": "<single string from X-axis categories>",
  "Y_Axis_Rubric_Category": "<single string from Y-axis categories>"
}}
```

## Assessment Process for Problem Statement Evaluation

### Step 1: Content Analysis (Y-Axis)
Carefully examine the problem statement to determine which of the following elements are clearly present. Select the ONE Y-axis category that includes all the identifiable elements:

**Contains Data**: Includes relevant quantitative or qualitative data supporting the existence of the problem.

**References Included**: Cites any sources or references.

**Location/Area Clear**: Clearly specifies the geographical location or specific area affected.

**Target Audience Clearly Stated**: Defines the specific group or demographic impacted by the issue.

**Impact Described**: Explains the consequences or negative outcomes if the problem remains unaddressed.

✔ Choose the category that matches all the elements present in the problem statement.

### Step 2: Quality Analysis (X-Axis) - ENHANCED WITH IDEA RELEVANCE
Evaluate the overall quality of the problem statement based on these five dimensions, with special attention to how well it aligns with the provided idea. Select the ONE best matching X-axis category based on how many of these are fully demonstrated:

**GRAMMAR**: Uses correct grammar, punctuation, spelling, sentence structure, and appropriate vocabulary throughout.

**DEMONSTRATES UNDERSTANDING**: Shows insight and clear comprehension of the topic and SDG concept.

**PRECISE AND TO THE POINT**: Avoids unnecessary detail and focuses on the core message without redundancy.

**RELEVANT TO THE IDEA**: Content clearly supports and aligns with the provided idea. The problem statement should logically connect to the idea as a potential solution or approach. If the problem statement is not relevant to the idea, this dimension is automatically NOT met.

**INFO IS WELL-STRUCTURED AND EASY TO UNDERSTAND**: Logical organization, clear flow, and easily comprehensible to readers.

✔ Choose the category that best reflects the overall writing quality and clarity based on the above dimensions, giving special weight to idea relevance.

### Critical Assessment Guidelines

**Idea Relevance Evaluation (Critical for X-Axis)**:
- Does the problem statement address the same issue that the idea is trying to solve?
- Are the problem statement and idea focused on the same or related SDG themes?
- Would the provided idea logically contribute to solving the stated problem?
- Is there thematic alignment between the problem context and the idea's scope?

**If the problem statement is NOT relevant to the provided idea**, the X-axis score should reflect this by selecting categories that include "Is not Relevant to the Idea" or lower quality combinations.

**Grammar Assessment (Enhanced)**:
Pay special attention to:
- Subject-verb agreement errors
- Incorrect tense usage
- Spelling mistakes
- Punctuation errors
- Run-on sentences or fragments
- Unclear or confusing sentence structure
- Inappropriate vocabulary for the context

## Critical Guidelines for Consistency

1. **Age-Appropriate Expectations**: Remember these are 14-16 year old students. Apply standards appropriate for this developmental level.

2. **Objective Analysis**: Base your assessment solely on what is explicitly present in the text, not on implied or inferred meanings.

3. **Consistent Criteria**: Use the same evaluation standards for every assessment. Apply consistent thresholds for grammar, understanding, precision, relevance to idea, and structure.

4. **Complete Analysis**: Examine ALL aspects thoroughly before making your final classification.

5. **Reproducible Results**: Your assessment of identical content must be identical every time, regardless of when the evaluation occurs.

6. **Idea Relevance Priority**: If the problem statement does not align with the provided idea, this significantly impacts the X-axis quality score.

## Valid Categories for Output

### X-AXIS CATEGORIES (Quality Assessment)

**Lowest Quality Level:**
- "Relevant to the Idea Only"
- "Does not have grammar"
- "Does not Demonstrate Understanding Only"
- "Is not Precise and To the Point"
- "Info is not Well-Structured and Is not Easy to Understand"

**Combined Low Quality:**
- "Does not have Grammar + Does not Demonstrate Understanding"
- "Does not have Grammar + Is not Precise and To the Point"
- "Does not have Grammar + Info is not Well-Structured and Is not Easy to Understand"
- "Does not Demonstrate Understanding + Info is not Well-Structured and Is not Easy to Understand"
- "Does not Demonstrate Understanding + Is not Precise and To the Point"
- "Is not Precise and To the Point + Info is not Well-Structured and Is not Easy to Understand"

**Triple and Quadruple Low Quality Combinations:**
- "Does not have Grammar + Does not Demonstrate Understanding + Is not Precise and To the Point"
- "Does not have Grammar + Does not Demonstrate Understanding + Info is not Well-Structured and Is not Easy to Understand"
- "Does not have Grammar + Is not Precise and To the Point + Info is not Well-Structured and Is not Easy to Understand"
- "Does not Demonstrate Understanding + Is not Precise and To the Point + Info is not Well-Structured and Is not Easy to Understand"
- "Does not have Grammar + Does not Demonstrate Understanding + Is not Precise and To the Point + Info is not Well-Structured and Is not Easy to Understand"

**Medium Quality Level:**
- "Has some grammar"
- "Demonstrates some Understanding"
- "Is somewhat Precise and To the Point"
- "Info is somewhat Well-Structured and fairly Easy to Understand"

**Combined Medium Quality:**
- "Has some Grammar + Demonstrates some Understanding"
- "Has some Grammar + is somewhat Precise and To the Point"
- "Has some Grammar + Info is somewhat Well-Structured and fairly Easy to Understand"
- "Demonstrates some Understanding + Info is somewhat Well-Structured and fairly Easy to Understand"
- "Demonstrates some Understanding + is somewhat Precise and To the Point"
- "Is somewhat Precise and To the Point + Info is somewhat Well-Structured and is somewhat Easy to Understand"

**Triple and Quadruple Medium Quality Combinations:**
- "Has some Grammar + Demonstrates some Understanding + is somewhat Precise and To the Point"
- "Has some Grammar + Demonstrates some Understanding + Info is somewhat Well-Structured and somewhat Easy to Understand"
- "Has some Grammar + is somewhat Precise and To the Point + Info is somewhat Well-Structured and is somewhat Easy to Understand"
- "Demonstrates some Understanding + is somewhat Precise and To the Point + Info is somewhat Well-Structured and is somewhat Easy to Understand"
- "Has some Grammar + Demonstrates some Understanding + is somewhat Precise and To the Point + Info is somewhat Well-Structured and somewhat Easy to Understand"

**High Quality Level:**
- "Has Very good Grammar + Demonstrates Very good Understanding"
- "Has Very good Grammar + Is Precise and To the Point"
- "Has Very good Grammar + Info is Well-Structured and Easy to Understand"
- "Demonstrates Very good Understanding + Is Precise and To the Point"
- "Demonstrates Very Good Understanding + Info is Very Well-Structured and Easy to Understand"
- "Is Precise and To the Point + Info is Very Well-Structured and Easy to Understand"

**Combined High Quality:**
- "Has very good Grammar + Demonstrates Very Good Understanding + Is Precise and To the Point"
- "Has Very Good Grammar + Demonstrates Very Good Understanding + Info is Very Well-Structured and Easy to Understand"
- "Has Very Good Grammar + Is Precise and To the Point + Info is Very Well-Structured and Easy to Understand"
- "Demonstrates Very Good Understanding + Is Precise and To the Point + Info is Very Well-Structured and Easy to Understand"
- "Has Very Good Grammar + Demonstrates Very Good Understanding + Is Precise and To the Point + Info is Very Well-Structured and Easy to Understand"

**Exceptional case:**
- "not Relevant content to the Idea"

### Y-AXIS CATEGORIES (Content Elements)

**Single Elements:**
- "Contains Data Only"
- "References Included Only"
- "Location/Area Clear Only"
- "Target Audience Clearly Stated Only"
- "Impact Described Only"

**Two Element Combinations:**
- "Contains Data + References Included"
- "Contains Data + Location/Area Clear"
- "Contains Data + Target Audience Clearly Stated"
- "Contains Data + Impact Described"
- "References Included + Location/Area Clear"
- "References Included + Target Audience Clearly Stated"
- "References Included + Impact Described"
- "Location/Area Clear + Target Audience Clearly Stated"
- "Location/Area Clear + Impact Described"
- "Target Audience Clearly Stated + Impact Described"

**Three Element Combinations:**
- "Contains Data + References Included + Location/Area Clear"
- "Contains Data + References Included + Target Audience Clearly Stated"
- "Contains Data + References Included + Impact Described"
- "Contains Data + Location/Area Clear + Target Audience Clearly Stated"
- "Contains Data + Location/Area Clear + Impact Described"
- "Contains Data + Target Audience Clearly Stated + Impact Described"
- "References Included + Location/Area Clear + Target Audience Clearly Stated"
- "References Included + Location/Area Clear + Impact Described"
- "References Included + Target Audience Clearly Stated + Impact Described"

**Four and Five Element Combinations:**
- "Contains Data + References Included + Location/Area Clear + Target Audience Clearly Stated"
- "Contains Data + References Included + Location/Area Clear + Impact Described"
- "Contains Data + References Included + Target Audience Clearly Stated + Impact Described"
- "Contains Data + References Included + Location/Area Clear + Target Audience Clearly Stated + Impact Described"

**Exceptional case:**
- "not Relevant content to the Idea"

---

Now analyze the following student submission:

**IDEA PROVIDED:**
{idea_text}

**PROBLEM STATEMENT TO EVALUATE:**
{problem_statement_text}

Carefully assess the problem statement's quality and content elements. Pay special attention to whether the problem statement is relevant to the provided idea. If they are not aligned or relevant to each other, this should significantly impact the X-axis quality score.

Provide ONLY the JSON output with the two required categories.
'''
    
    def _validate_inputs(self, idea_text: str, problem_statement_text: str) -> None:
        """Validate input parameters with optimized checks."""
        if not idea_text or not isinstance(idea_text, str) or len(idea_text.strip()) < 10:
            raise ValidationError("idea_text must be a non-empty string with at least 10 characters")
        
        if not problem_statement_text or not isinstance(problem_statement_text, str) or len(problem_statement_text.strip()) < 20:
            raise ValidationError("problem_statement_text must be a non-empty string with at least 20 characters")
        
        # Quick length checks
        if len(idea_text) > 3000:
            raise ValidationError("idea_text exceeds maximum length of 3000 characters")
        
        if len(problem_statement_text) > 8000:
            raise ValidationError("problem_statement_text exceeds maximum length of 8000 characters")
    
    def _sanitize_text(self, text: str) -> str:
        """Fast text sanitization."""
        return ' '.join(text.strip().split())
    
    def _create_cache_key(self, idea_text: str, problem_statement_text: str) -> str:
        """Create optimized cache key."""
        combined = f"{idea_text[:200]}|{problem_statement_text[:500]}"  # Limit key size
        return hashlib.blake2b(combined.encode(), digest_size=16).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self._cache_timestamps:
            return False
        
        return (time.time() - self._cache_timestamps[cache_key]) < self.cache_ttl
    
    def _get_cached_result(self, cache_key: str) -> Optional[ClassificationResult]:
        """Get cached result with TTL check."""
        if not self.enable_caching or cache_key not in self._cache:
            self.metrics['cache_misses'] += 1
            return None
        
        if not self._is_cache_valid(cache_key):
            # Remove expired cache entry
            del self._cache[cache_key]
            del self._cache_timestamps[cache_key]
            self.metrics['cache_misses'] += 1
            return None
        
        self.metrics['cache_hits'] += 1
        return self._cache[cache_key]
    
    def _cache_result(self, cache_key: str, result: ClassificationResult) -> None:
        """Cache result with TTL."""
        if not self.enable_caching:
            return
        
        # Simple LRU eviction
        if len(self._cache) >= self.cache_size:
            oldest_key = min(self._cache_timestamps.keys(), 
                           key=self._cache_timestamps.get)
            del self._cache[oldest_key]
            del self._cache_timestamps[oldest_key]
        
        self._cache[cache_key] = result
        self._cache_timestamps[cache_key] = time.time()
    
    def _parse_json_response(self, response_text: str) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Fast JSON parsing with fallback."""
        try:
            # Clean response
            text = response_text.strip()
            
            # Remove markdown formatting
            if text.startswith('```json'):
                text = text[7:]
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()
            
            # Find JSON boundaries
            start = text.find('{')
            end = text.rfind('}') + 1
            
            if start == -1 or end <= start:
                return False, {}, "No JSON found in response"
            
            json_str = text[start:end]
            parsed = json.loads(json_str)
            
            # Quick validation
            required_keys = ["X_Axis_Rubric_Category", "Y_Axis_Rubric_Category"]
            if not all(k in parsed and isinstance(parsed[k], str) and parsed[k].strip() 
                      for k in required_keys):
                return False, {}, "Invalid or missing required fields"
            
            return True, parsed, None
            
        except json.JSONDecodeError as e:
            return False, {}, f"JSON parsing error: {str(e)}"
        except Exception as e:
            return False, {}, f"Response parsing error: {str(e)}"
    
    async def _make_api_call_async(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Make async API call to OpenAI."""
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a precise assessment agent. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low for consistency
                max_tokens=1000,
                response_format={"type": "json_object"},
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            usage_info = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            return response.choices[0].message.content, usage_info
            
        except Exception as e:
            raise APIError(f"OpenAI API call failed: {str(e)}")
    
    def _make_api_call_sync(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Make synchronous API call to OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a precise assessment agent. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"},
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            usage_info = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            return response.choices[0].message.content, usage_info
            
        except Exception as e:
            raise APIError(f"Groq API call failed: {str(e)}")
    
    async def classify_async(self, idea_text: str, problem_statement_text: str) -> ClassificationResult:
        """Async classification method for maximum speed."""
        start_time = time.time()
        self.metrics['total_requests'] += 1
        
        try:
            # Fast validation and sanitization
            self._validate_inputs(idea_text, problem_statement_text)
            idea_text = self._sanitize_text(idea_text)
            problem_statement_text = self._sanitize_text(problem_statement_text)
            
            # Check cache
            cache_key = self._create_cache_key(idea_text, problem_statement_text)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # Build optimized prompt
            prompt = self._get_optimized_prompt().format(
                idea_text=idea_text,
                problem_statement_text=problem_statement_text
            )
            
            # Make API call with retries
            last_error = None
            for attempt in range(self.max_retries):
                try:
                    response_text, usage_info = await self._make_api_call_async(prompt)
                    
                    # Parse response
                    success, parsed_data, error = self._parse_json_response(response_text)
                    
                    if success:
                        processing_time = time.time() - start_time
                        
                        result = ClassificationResult(
                            success=True,
                            x_axis_category=parsed_data['X_Axis_Rubric_Category'],
                            y_axis_category=parsed_data['Y_Axis_Rubric_Category'],
                            raw_response=response_text,
                            processing_time=processing_time,
                            tokens_used=usage_info.get('total_tokens', 0),
                            model_used=self.model_name
                        )
                        
                        # Update metrics
                        self._update_metrics(processing_time, usage_info.get('total_tokens', 0), True)
                        
                        # Cache result
                        self._cache_result(cache_key, result)
                        
                        return result
                    else:
                        raise APIError(f"Response parsing failed: {error}")
                
                except Exception as e:
                    last_error = e
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(0.5 * (attempt + 1))  # Fast exponential backoff
            
            # All retries failed
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, 0, False)
            
            return ClassificationResult(
                success=False,
                error=f"Failed after {self.max_retries} attempts: {str(last_error)}",
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, 0, False)
            
            return ClassificationResult(
                success=False,
                error=str(e),
                processing_time=processing_time
            )
    
    def classify_sync(self, idea_text: str, problem_statement_text: str) -> ClassificationResult:
        """Synchronous classification method."""
        return asyncio.run(self.classify_async(idea_text, problem_statement_text))
    
    # Alias for backwards compatibility
    def classify_problem_statement(self, idea_text: str, problem_statement_text: str) -> ClassificationResult:
        """Main classification method (backwards compatible)."""
        return self.classify_sync(idea_text, problem_statement_text)
    
    async def classify_batch_async(self, requests: List[Tuple[str, str]]) -> List[ClassificationResult]:
        """High-speed batch classification."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def classify_single(idea: str, problem: str) -> ClassificationResult:
            async with semaphore:
                return await self.classify_async(idea, problem)
        
        tasks = [classify_single(idea, problem) for idea, problem in requests]
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    def classify_batch_sync(self, requests: List[Tuple[str, str]]) -> List[ClassificationResult]:
        """Synchronous batch classification."""
        return asyncio.run(self.classify_batch_async(requests))
    
    def _update_metrics(self, processing_time: float, tokens: int, success: bool) -> None:
        """Update performance metrics."""
        if success:
            self.metrics['successful_requests'] += 1
            
            # Update response time stats
            if self.metrics['successful_requests'] == 1:
                self.metrics['average_response_time'] = processing_time
            else:
                n = self.metrics['successful_requests']
                current_avg = self.metrics['average_response_time']
                self.metrics['average_response_time'] = (current_avg * (n - 1) + processing_time) / n
            
            # Update fastest/slowest
            self.metrics['fastest_response'] = min(self.metrics['fastest_response'], processing_time)
            self.metrics['slowest_response'] = max(self.metrics['slowest_response'], processing_time)
            
            # Update token usage
            self.metrics['total_tokens'] += tokens
        else:
            self.metrics['failed_requests'] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        total_requests = self.metrics['total_requests']
        if total_requests == 0:
            return {"message": "No requests processed yet"}
        
        success_rate = (self.metrics['successful_requests'] / total_requests) * 100
        cache_hit_rate = (self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses'])) * 100 if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0
        
        return {
            'total_requests': total_requests,
            'successful_requests': self.metrics['successful_requests'],
            'failed_requests': self.metrics['failed_requests'],
            'success_rate': f"{success_rate:.2f}%",
            'cache_hits': self.metrics['cache_hits'],
            'cache_misses': self.metrics['cache_misses'],
            'cache_hit_rate': f"{cache_hit_rate:.2f}%",
            'average_response_time': f"{self.metrics['average_response_time']:.3f}s",
            'fastest_response': f"{self.metrics['fastest_response']:.3f}s" if self.metrics['fastest_response'] != float('inf') else "N/A",
            'slowest_response': f"{self.metrics['slowest_response']:.3f}s",
            'total_tokens_used': self.metrics['total_tokens'],
            'model_used': self.model_name,
            'cache_size': len(self._cache)
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_response_time': 0.0,
            'total_tokens': 0,
            'fastest_response': float('inf'),
            'slowest_response': 0.0
        }
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        self._cache_timestamps.clear()
        self.logger.info("Cache cleared")
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Factory functions
def create_fast_classifier(api_key: Optional[str] = None, **kwargs) -> OptimizedProblemStatementClassifier:
    """Create a classifier optimized for speed."""
    defaults = {
        'model_name': 'gpt-4.1',  # Only available model
        'max_retries': 2,
        'timeout': 10.0,
        'cache_size': 512,
        'max_concurrent': 15
    }
    defaults.update(kwargs)
    return OptimizedProblemStatementClassifier(api_key=api_key, **defaults)


def create_balanced_classifier(api_key: Optional[str] = None, **kwargs) -> OptimizedProblemStatementClassifier:
    """Create a classifier with standard settings."""
    defaults = {
        'model_name': 'gpt-4.1',  # Only available model
        'max_retries': 2,
        'timeout': 15.0,
        'cache_size': 256,
        'max_concurrent': 10
    }
    defaults.update(kwargs)
    return OptimizedProblemStatementClassifier(api_key=api_key, **defaults)


# Test samples
SAMPLE_DATA = [
    {
        'idea': 'Smart Water Management System: IoT-enabled sensors and mobile app for monitoring water quality and usage in rural communities, promoting SDG 6 (Clean Water and Sanitation).',
        'problem': 'Water scarcity in rural Maharashtra affects over 2.5 million people according to state government data from 2023. Villages like Ahmednagar face daily water shortages during summer months (April-June), forcing women and children to walk 3-5 kilometers daily to collect water. This impacts school attendance rates, which drop by 35% during peak summer, particularly affecting girls aged 10-14. Without proper monitoring systems, existing bore wells often run dry or produce contaminated water, leading to waterborne diseases that affect approximately 40% of households annually.'
    },
    {
        'idea': 'Community Recycling Hub: Local waste sorting and recycling center using AI-powered sorting technology to promote circular economy and SDG 12 (Responsible Consumption).',
        'problem': 'Urban waste management crisis in Delhi generates 11,000 tons of waste daily but only processes 61% effectively. Most residential areas lack proper segregation systems, leading to 85% mixed waste that ends up in landfills. This creates methane emissions equivalent to 2.3 million tons CO2 annually and affects air quality for 30 million residents. Local recyclers struggle with contaminated materials, reducing recycling efficiency to just 12% compared to global averages of 35%.'
    },
    {
        'idea': 'Digital Learning Platform for Girls: Mobile education app providing STEM courses in local languages to promote girls\' education and SDG 4 (Quality Education).',
        'problem': 'Gender disparity in STEM education affects millions of girls globally, with only 35% of STEM higher education students being female according to UNESCO data.'
    }
]


async def run_comprehensive_test():
    """Run comprehensive tests with sample data."""
    print("🚀 Testing Optimized Problem Statement Classifier with Groq API")
    print("=" * 70)
    
    # Create classifier instance
    try:
        classifier = create_balanced_classifier(log_level="INFO")
        print(f"✅ Classifier created successfully with model: {classifier.model_name}")
    except Exception as e:
        print(f"❌ Failed to create classifier: {e}")
        return
    
    # Test 1: Single classification
    print("\n📋 Test 1: Single Classification")
    print("-" * 40)
    
    sample = SAMPLE_DATA[0]
    start_time = time.time()
    
    result = await classifier.classify_async(sample['idea'], sample['problem'])
    
    if result.success:
        print("✅ Classification successful!")
        print(f"   Quality (X-Axis): {result.x_axis_category}")
        print(f"   Content (Y-Axis): {result.y_axis_category}")
        print(f"   Processing time: {result.processing_time:.3f}s")
        print(f"   Tokens used: {result.tokens_used}")
        print(f"   Model: {result.model_used}")
    else:
        print(f"❌ Classification failed: {result.error}")
    
    # Test 2: Batch classification
    print(f"\n📦 Test 2: Batch Classification ({len(SAMPLE_DATA)} samples)")
    print("-" * 50)
    
    batch_requests = [(sample['idea'], sample['problem']) for sample in SAMPLE_DATA]
    batch_start = time.time()
    
    batch_results = await classifier.classify_batch_async(batch_requests)
    batch_time = time.time() - batch_start
    
    print(f"⏱️  Total batch time: {batch_time:.3f}s")
    print(f"📊 Results:")
    
    for i, result in enumerate(batch_results):
        if result.success:
            print(f"   Sample {i+1}: ✅ X: {result.x_axis_category[:30]}...")
            print(f"              Y: {result.y_axis_category[:30]}...")
            print(f"              Time: {result.processing_time:.3f}s")
        else:
            print(f"   Sample {i+1}: ❌ Error: {result.error}")
    
    # Test 3: Cache performance
    print(f"\n🔄 Test 3: Cache Performance Test")
    print("-" * 40)
    
    # Run same request twice to test caching
    sample = SAMPLE_DATA[1]
    
    # First call (cache miss)
    cache_start = time.time()
    result1 = await classifier.classify_async(sample['idea'], sample['problem'])
    first_time = time.time() - cache_start
    
    # Second call (should be cache hit)
    cache_start = time.time()
    result2 = await classifier.classify_async(sample['idea'], sample['problem'])
    second_time = time.time() - cache_start
    
    print(f"   First call (cache miss): {first_time:.3f}s")
    print(f"   Second call (cache hit): {second_time:.3f}s")
    print(f"   Speed improvement: {(first_time/second_time):.1f}x faster")
    print(f"   Results identical: {result1.x_axis_category == result2.x_axis_category and result1.y_axis_category == result2.y_axis_category}")
    
    # Test 4: Performance report
    print(f"\n📈 Performance Report")
    print("-" * 30)
    
    report = classifier.get_performance_report()
    for key, value in report.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n🎉 Testing completed successfully!")
    return classifier


def sync_test():
    """Synchronous test runner."""
    print("Running synchronous tests...")
    
    try:
        classifier = create_fast_classifier()
        
        # Quick sync test
        sample = SAMPLE_DATA[0]
        result = classifier.classify_sync(sample['idea'], sample['problem'])
        
        if result.success:
            print("✅ Sync test passed!")
            print(f"   Processing time: {result.processing_time:.3f}s")
            print(f"   X-Axis: {result.x_axis_category}")
            print(f"   Y-Axis: {result.y_axis_category}")
        else:
            print(f"❌ Sync test failed: {result.error}")
        
        print(f"\nPerformance: {classifier.get_performance_report()}")
        
    except Exception as e:
        print(f"❌ Sync test error: {e}")


if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Async comprehensive test (recommended)")
    print("2. Quick sync test")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(run_comprehensive_test())
    else:
        sync_test()