import streamlit as st
import json
import logging
import os
import sys
import base64
from typing import Dict, List, Optional, Any
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sdg_app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules with proper error handling
def safe_import(module_name: str, function_name: str = None):
    """Safely import modules and functions with proper error handling"""
    try:
        module = __import__(module_name)
        if function_name:
            return getattr(module, function_name)
        return module
    except ImportError as e:
        logger.error(f"Failed to import {module_name}: {str(e)}")
        return None
    except AttributeError as e:
        logger.error(f"Function {function_name} not found in {module_name}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error importing {module_name}: {str(e)}")
        return None

# Import functions
generate_sdg_ideas_json = safe_import('generate_idea', 'generate_sdg_ideas_json')
OptimizedProblemStatementClassifier = safe_import('evaluate_problem_statement', 'OptimizedProblemStatementClassifier')
generate_market_insights = safe_import('market_research', 'generate_market_insights')
verify_env_setup = safe_import('market_research', 'verify_env_setup')
generate_survey_questions = safe_import('generate_questions', 'generate_survey_questions')
get_market_fit_feedback = safe_import('generate_marketfit_feedback', 'get_market_fit_feedback')

# Page configuration
st.set_page_config(
    page_title="SDG Innovation Hub",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .step-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2a5298;
        margin-bottom: 1rem;
    }
    .success-card {
        background: #d4edda;
        border-color: #28a745;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-card {
        background: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stProgress .st-bo {
        background-color: #2a5298;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with default values
def initialize_session_state():
    """Initialize all session state variables with default values"""
    default_values = {
        'page': 1,
        'selected_sdgs': [],
        'generated_ideas': [],
        'selected_idea': "",
        'problem_statement': "",
        'problem_statement_classification_result': None,
        'target_market': "",
        'market_summary': "",
        'research_completed': False,
        'research_questions': [],
        'survey_questions': [],
        'survey_responses': {},
        'marketfit': "",
        'market_fit_feedback': "",
        'market_research_results': None,
        'market_research_evaluated': False,
        'categorization_result': None,
        'prototype_description': "",
        'prototype_images': [],
        'prototype_prompt': "",
        'prototype_style': "",
        'prototype_generated': False,
        'prototype_evaluation': None,
        'market_fit_evaluation': None,
        'app_initialized': True
    }
    
    for key, default_value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize session state
initialize_session_state()

# SDG List with full descriptions
SDG_LIST = [
    "1. No Poverty",
    "2. Zero Hunger", 
    "3. Good Health and Well-being",
    "4. Quality Education",
    "5. Gender Equality",
    "6. Clean Water and Sanitation",
    "7. Affordable and Clean Energy",
    "8. Decent Work and Economic Growth",
    "9. Industry, Innovation and Infrastructure",
    "10. Reduced Inequalities",
    "11. Sustainable Cities and Communities",
    "12. Responsible Consumption and Production",
    "13. Climate Action",
    "14. Life Below Water",
    "15. Life on Land",
    "16. Peace, Justice and Strong Institutions",
    "17. Partnerships for the Goals"
]

# OpenAI Configuration
OPENAI_MODEL = "gpt-4.1"  # Updated to GPT-4.1

def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from environment variables or Streamlit secrets"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            pass
    return api_key

def validate_api_key() -> bool:
    """Validate if OpenAI API key is available"""
    api_key = get_openai_api_key()
    if not api_key:
        st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY in your environment variables or Streamlit secrets.")
        return False
    return True

# Navigation functions
def navigate_to_page(page_num: int):
    """Navigate to a specific page"""
    st.session_state.page = page_num
    logger.info(f"Navigated to page {page_num}")

def can_navigate_to_page(page_num: int) -> bool:
    """Check if user can navigate to a specific page based on completed steps"""
    if page_num == 1:
        return True
    elif page_num == 2:
        return bool(st.session_state.selected_idea)
    elif page_num == 3:
        return bool(st.session_state.problem_statement)
    elif page_num == 4:
        return bool(st.session_state.target_market and st.session_state.market_summary)
    elif page_num == 5:
        return bool(st.session_state.survey_questions)
    elif page_num == 6:
        return bool(st.session_state.marketfit)
    return False

# Sidebar navigation
def render_sidebar():
    """Render the sidebar navigation"""
    st.sidebar.title("🌍 SDG Innovation Hub")
    st.sidebar.markdown("---")
    
    # Progress indicator
    progress = st.session_state.page / 6
    st.sidebar.progress(progress, f"Step {st.session_state.page} of 6")
    
    st.sidebar.markdown("### Navigation")
    
    pages = [
        ("1. SDG Selection & Ideas", "🎯"),
        ("2. Problem Statement", "📝"), 
        ("3. Market Research", "🔍"),
        ("4. Survey Questions", "❓"),
        ("5. Market Fit Analysis", "📈"),
        ("6. Prototype Generation", "🚀")
    ]
    
    for i, (page_name, icon) in enumerate(pages, 1):
        disabled = not can_navigate_to_page(i)
        if st.sidebar.button(
            f"{icon} {page_name}", 
            key=f"nav_{i}",
            disabled=disabled,
            use_container_width=True
        ):
            navigate_to_page(i)
            st.rerun()
    
    # Show current session info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Session Info")
    if st.session_state.selected_sdgs:
        st.sidebar.write(f"**SDGs:** {len(st.session_state.selected_sdgs)} selected")
    if st.session_state.selected_idea:
        st.sidebar.write("**Idea:** ✅ Selected")
    if st.session_state.problem_statement:
        st.sidebar.write("**Problem:** ✅ Defined")
    if st.session_state.target_market:
        st.sidebar.write("**Market:** ✅ Researched")
    
    # Environment check
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    api_key_status = "✅" if get_openai_api_key() else "❌"
    st.sidebar.write(f"**OpenAI API:** {api_key_status}")

# Page 1: SDG Selection and Idea Generation
def render_page_1():
    """Render SDG selection and idea generation page"""
    st.markdown('<div class="main-header"><h1>🌍 SDG Innovation Hub</h1><p>Transform Global Challenges into Innovation Opportunities</p></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.header("Step 1: Select SDGs and Generate Ideas")
    st.markdown("Choose exactly 2 Sustainable Development Goals to focus your innovation efforts.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # SDG Selection
    st.subheader("🎯 Choose 2 Sustainable Development Goals")
    
    selected_sdgs = []
    
    # Display SDGs in 3 columns for better layout
    col1, col2, col3 = st.columns(3)
    
    for i, sdg in enumerate(SDG_LIST):
        col = [col1, col2, col3][i % 3]
        with col:
            if st.checkbox(sdg, key=f"sdg_check_{i}"):
                selected_sdgs.append(sdg)
    
    # Validation
    if len(selected_sdgs) > 2:
        st.error("⚠️ Please select exactly 2 SDGs. You have selected too many.")
        st.session_state.selected_sdgs = []
    elif len(selected_sdgs) == 2:
        st.session_state.selected_sdgs = selected_sdgs
        st.success(f"✅ Selected SDGs: {selected_sdgs[0]} and {selected_sdgs[1]}")
        
        # Idea Generation
        if st.button("🚀 Generate Ideas", type="primary", use_container_width=True):
            if not validate_api_key():
                return
                
            if generate_sdg_ideas_json is not None:
                try:
                    with st.spinner("🤖 Generating innovative ideas..."):
                        result = generate_sdg_ideas_json(selected_sdgs)
                        if result:
                            st.session_state.generated_ideas = []
                            for idea in result:
                                title = idea.get("title", "Untitled")
                                description = idea.get("description", "No description provided.")
                                st.session_state.generated_ideas.append(f"**{title}**\n\n{description}")
                            st.success(f"✅ Generated {len(result)} innovative ideas!")
                            logger.info(f"Generated {len(result)} ideas for SDGs: {selected_sdgs}")
                        else:
                            st.error("❌ Failed to generate ideas. Please try again.")
                except Exception as e:
                    st.error(f"❌ Error generating ideas: {str(e)}")
                    logger.error(f"Error in generate_sdg_ideas_json: {str(e)}\n{traceback.format_exc()}")
            else:
                st.error("❌ Idea generation module not available. Please check imports.")
            st.rerun()
            
    elif len(selected_sdgs) == 1:
        st.info("📋 Please select one more SDG to proceed.")
        st.session_state.selected_sdgs = []
    else:
        st.info("📋 Please select 2 SDGs to get started.")
        st.session_state.selected_sdgs = []
    
    # Display generated ideas
    if st.session_state.generated_ideas:
        st.markdown("---")
        st.subheader("💡 Generated Ideas")
        
        selected_idea = st.radio(
            "Select one idea to proceed:",
            st.session_state.generated_ideas,
            key="idea_selection"
        )
        
        st.session_state.selected_idea = selected_idea
        
        if st.button("Proceed to Problem Statement →", type="primary", use_container_width=True):
            navigate_to_page(2)
            st.rerun()

# Page 2: Problem Statement
def render_page_2():
    """Render problem statement definition page"""
    st.title("📝 Problem Statement")
    st.header("Step 2: Define the Problem")
    
    if st.session_state.selected_idea:
        st.info(f"**Selected Idea:** {st.session_state.selected_idea}")
    
    st.subheader("✍️ Write Your Problem Statement")
    st.markdown("Clearly articulate the specific problem your idea addresses. A good problem statement is specific, measurable, and actionable.")
    
    problem_statement = st.text_area(
        "Describe the problem your idea addresses:",
        value=st.session_state.problem_statement,
        height=150,
        placeholder="Enter a detailed problem statement here...",
        key="problem_statement_input",
        help="Include: What is the problem? Who does it affect? Why is it important to solve?"
    )
    
    st.session_state.problem_statement = problem_statement
    
    # Character count
    char_count = len(problem_statement)
    st.caption(f"Characters: {char_count} (Recommended: 100-500)")
    
    if problem_statement and char_count >= 50:
        if st.button("🔍 Evaluate Problem Statement", type="primary", use_container_width=True):
            if not validate_api_key():
                return
                
            if OptimizedProblemStatementClassifier is not None:
                try:
                    with st.spinner("🤖 Evaluating problem statement..."):
                        classifier = OptimizedProblemStatementClassifier(
                            api_key=get_openai_api_key(),
                            model_name=OPENAI_MODEL,
                            max_retries=3,
                            timeout=30.0,
                            enable_caching=True
                        )
                        
                        result = classifier.classify_problem_statement(
                            idea_text=st.session_state.selected_idea,
                            problem_statement_text=st.session_state.problem_statement
                        )
                        
                        if result and hasattr(result, 'success') and result.success:
                            st.success("✅ Problem statement evaluated successfully!")
                            
                            col1, col2 = st.columns(2)

                            # with col1:
                            #     st.metric("Quality Assessment", getattr(result, 'x_axis_category', 'N/A'))

                            # with col2:
                            #     st.metric("Content Assessment", getattr(result, 'y_axis_category', 'N/A'))

                            with st.expander("🔍 Detailed Evaluation", expanded=True):
                                st.markdown("**🧠 X-Axis (Quality Assessment):**")
                                st.write(getattr(result, 'x_axis_category', 'N/A'))

                                st.markdown("**📘 Y-Axis (Content Assessment):**")
                                st.write(getattr(result, 'y_axis_category', 'N/A'))

                                                            
                            st.session_state.problem_statement_classification_result = result
                            logger.info("Problem statement evaluation completed successfully")
                            
                        else:
                            error_msg = getattr(result, 'error', 'Unknown error') if result else 'No result returned'
                            st.error(f"❌ Evaluation failed: {error_msg}")
                            
                except Exception as e:
                    st.error(f"❌ Classifier error: {str(e)}")
                    logger.error(f"Classifier error: {str(e)}\n{traceback.format_exc()}")
            else:
                st.error("❌ Problem statement classifier not available. Please check imports.")
    elif problem_statement and char_count < 50:
        st.warning("⚠️ Please provide a more detailed problem statement (at least 50 characters).")
    
    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Back to SDG Selection", type="secondary", use_container_width=True):
            navigate_to_page(1)
            st.rerun()
    
    with col2:
        if st.session_state.problem_statement_classification_result:
            if st.button("Proceed to Market Research →", type="primary", use_container_width=True):
                navigate_to_page(3)
                st.rerun()
        else:
            st.button("Proceed to Market Research →", disabled=True, help="Evaluate problem statement first")

# Page 3: Market Research
def render_page_3():
    """Render market research page"""
    st.title("🎯 Market Research")
    st.header("Step 3: Define Target Market and Research")
    
    if st.session_state.selected_idea:
        st.info(f"**Selected Idea:** {st.session_state.selected_idea}")
    
    # Target Market Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Target Market")
        target_market = st.text_area(
            "Define your target market:",
            value=st.session_state.target_market,
            height=120,
            placeholder="Demographics, geography, psychographics, market size...",
            key="target_market_input",
            help="Who are your potential customers? Be specific about age, location, income, behaviors, etc."
        )
        st.caption(f"Characters: {len(target_market)}/400")
    
    with col2:
        st.subheader("📚 Existing Knowledge")
        market_summary = st.text_area(
            "What do you already know?",
            value=st.session_state.market_summary,
            height=120,
            placeholder="Existing research, competitors, market trends...",
            key="market_summary_input",
            help="Describe existing knowledge about competitors, market size, trends, etc."
        )
        st.caption(f"Characters: {len(market_summary)}/400")
    
    # Update session state
    st.session_state.target_market = target_market
    st.session_state.market_summary = market_summary
    
    # Validation
    basic_fields_filled = all([
        target_market.strip(), 
        market_summary.strip(), 
        st.session_state.problem_statement, 
        st.session_state.selected_idea
    ])
    
    char_limit_ok = len(target_market) <= 400 and len(market_summary) <= 400
    
    if not char_limit_ok:
        st.error("❌ Target market and market summary must be under 400 characters each.")
    
    if not basic_fields_filled:
        st.warning("⚠️ Please fill in all fields before proceeding.")
    
    # Market Research Generation
    if st.button("🔬 Generate Market Research", type="primary", use_container_width=True):
        if not validate_api_key():
            return
            
        if basic_fields_filled and char_limit_ok:
            with st.spinner("🔍 Conducting comprehensive market research..."):
                try:
                    results = generate_market_insights(
                        target_market=target_market,
                        market_summary=market_summary,
                        idea=st.session_state.selected_idea,
                        problem_statement=st.session_state.problem_statement
                    )
                    
                    if "error" in results:
                        st.error(f"❌ Error: {results['error']}")
                        if "instructions" in results:
                            st.info(results['instructions'])
                    else:
                        # Display results
                        st.success("✅ Market research completed successfully!")
                        
                        # Consistency check
                        if "consistency_check" in results:
                            if "✅" in results['consistency_check']:
                                st.success(f"📋 Consistency: {results['consistency_check']}")
                            else:
                                st.warning(f"📋 Consistency: {results['consistency_check']}")
                        
                        # Market analysis
                        if "market_analysis" in results:
                            st.subheader("📊 Market Analysis")
                            st.markdown(results['market_analysis'])
                        
                        # Sources
                        if results.get('web_sources'):
                            with st.expander("🔗 Research Sources", expanded=False):
                                for i, source in enumerate(results['web_sources'], 1):
                                    st.markdown(f"**Source {i}: {source['title']}**")
                                    st.markdown(source['content'][:500] + "..." if len(source['content']) > 500 else source['content'])
                                    st.markdown(f"*URL: {source['url']}*")
                                    st.markdown("---")
                        
                        st.session_state.market_research_results = results
                        logger.info("Market research completed successfully")
                        
                except Exception as e:
                    st.error(f"❌ Error generating market research: {str(e)}")
                    logger.error(f"Market research error: {str(e)}\n{traceback.format_exc()}")
    
    # Market Research Evaluation
    if hasattr(st.session_state, 'market_research_results') and st.session_state.market_research_results:
        st.markdown("---")
        
        if st.button("📋 Evaluate Market Research", type="secondary", use_container_width=True):
            with st.spinner("🔍 Evaluating market research quality..."):
                try:
                    categorize_market_research = safe_import('evaluate_market_research', 'categorize_market_research')
                    if categorize_market_research:
                        data = f"{market_summary} {target_market}"
                        result = categorize_market_research(data)
                        
                        st.session_state.market_research_evaluated = True
                        st.session_state.categorization_result = result
                        
                        st.success("✅ Market research evaluation completed!")
                        
                        with st.expander("🔍 View Evaluation Results"):
                            st.json(result)
                        
                        logger.info("Market research evaluation completed")
                    else:
                        st.error("❌ Evaluation module not available")
                        
                except Exception as e:
                    st.error(f"❌ Error evaluating market research: {str(e)}")
                    logger.error(f"Market research evaluation error: {str(e)}")
    
    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Back to Problem Statement", type="secondary", use_container_width=True):
            navigate_to_page(2)
            st.rerun()
    
    with col2:
        if basic_fields_filled and char_limit_ok:
            if st.button("Proceed to Survey Questions →", type="primary", use_container_width=True):
                navigate_to_page(4)
                st.rerun()
        else:
            st.button("Proceed to Survey Questions →", disabled=True, help="Complete all fields first")

# Page 4: Survey Questions
def render_page_4():
    """Render survey questions generation page"""
    st.title("❓ Survey Questions")
    st.header("Step 4: Generate Market Validation Questions")
    
    # Display context
    if st.session_state.selected_idea and st.session_state.problem_statement:
        with st.expander("📋 Project Context", expanded=False):
            st.write(f"**Selected SDGs:** {', '.join(st.session_state.selected_sdgs)}")
            st.write(f"**Idea:** {st.session_state.selected_idea}")
            st.write(f"**Problem:** {st.session_state.problem_statement}")
        
        # Generate Questions
        if st.button("📋 Generate Survey Questions", type="primary", use_container_width=True):
            if not validate_api_key():
                return
                
            with st.spinner("🤖 Generating market validation questions..."):
                try:
                    result = generate_survey_questions(
                        idea=st.session_state.selected_idea,
                        problem_statement=st.session_state.problem_statement,
                        sdgs=st.session_state.selected_sdgs
                    )
                    
                    st.session_state.survey_questions = result
                    if 'survey_responses' not in st.session_state:
                        st.session_state.survey_responses = {}
                    
                    st.success(f"✅ Generated {len(result)} survey questions!")
                    logger.info(f"Generated {len(result)} survey questions")
                    
                except Exception as e:
                    st.error(f"❌ Error generating questions: {str(e)}")
                    logger.error(f"Survey generation error: {str(e)}\n{traceback.format_exc()}")
        
        # Display Questions
        if st.session_state.survey_questions:
            st.markdown("---")
            st.subheader("📝 Market Validation Questions")
            
            if 'survey_responses' not in st.session_state:
                st.session_state.survey_responses = {}
            
            # Display questions with response options
            for i, question in enumerate(st.session_state.survey_questions, 1):
                with st.container():
                    st.markdown(f"**{i}.** {question}")
                    
                    response = st.radio(
                        f"Response for Question {i}", 
                        ["👍 Yes", "👎 No"], 
                        key=f"response_{i}",
                        index=None,
                        horizontal=True
                    )
                    
                    if response:
                        st.session_state.survey_responses[f"question_{i}"] = response
                    
                    st.markdown("---")
            
            # Response Summary
            if st.session_state.survey_responses:
                completed = len(st.session_state.survey_responses)
                total = len(st.session_state.survey_questions)
                
                st.subheader("📊 Response Summary")
                st.progress(completed / total, f"Completed: {completed}/{total} questions")
                
                # Show response breakdown
                yes_count = sum(1 for response in st.session_state.survey_responses.values() if "Yes" in response)
                no_count = completed - yes_count
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Yes Responses", yes_count)
                with col2:
                    st.metric("No Responses", no_count)
                with col3:
                    st.metric("Completion Rate", f"{(completed/total)*100:.0f}%")
    
    else:
        st.error("❌ Missing required information. Please complete previous steps.")
        st.markdown("**Required:**")
        st.markdown("- Selected SDGs ✅" if st.session_state.selected_sdgs else "- Selected SDGs ❌")
        st.markdown("- Selected Idea ✅" if st.session_state.selected_idea else "- Selected Idea ❌")
        st.markdown("- Problem Statement ✅" if st.session_state.problem_statement else "- Problem Statement ❌")
    
    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Back to Market Research", type="secondary", use_container_width=True):
            navigate_to_page(3)
            st.rerun()
    
    with col2:
        if st.session_state.survey_questions:
            if st.button("Proceed to Market Fit Analysis →", type="primary", use_container_width=True):
                navigate_to_page(5)
                st.rerun()
        else:
            st.button("Proceed to Market Fit Analysis →", disabled=True, help="Generate survey questions first")

# Page 5: Market Fit Analysis
def render_page_5():
    """Render market fit analysis page"""
    st.title("📈 Market Fit Analysis")
    st.header("Step 5: Analyze Market-Solution Fit")
    
    # Project Context
    if st.session_state.selected_idea:
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**SDGs:** {', '.join(st.session_state.selected_sdgs)}")
            st.info(f"**Idea:** {st.session_state.selected_idea[:100]}...")
        
        with col2:
            st.info(f"**Problem:** {st.session_state.problem_statement[:100]}...")
        
        # Expandable sections for detailed info
        with st.expander("📊 Survey Data", expanded=False):
            if st.session_state.survey_responses:
                for key, value in st.session_state.survey_responses.items():
                    question_num = key.split('_')[1]
                    st.write(f"**Q{question_num}:** {value}")
            else:
                st.write("No survey responses available")
        
        with st.expander("🔍 Market Research", expanded=False):
            if st.session_state.market_research_results:
                st.write(st.session_state.market_research_results.get('market_analysis', 'No analysis available'))
            else:
                st.write("No market research available")
    
    # Market Fit Analysis Input
    st.subheader("📝 Market Fit Analysis")
    st.markdown("Provide a comprehensive analysis of how well your solution fits the target market:")
    
    marketfit = st.text_area(
        "Your Market Fit Analysis:",
        value=st.session_state.get('marketfit', ''),
        height=200,
        placeholder="""Consider analyzing:
• Target market size and accessibility
• User feedback and validation results
• Competitive landscape and differentiation
• Value proposition alignment with market needs
• Pricing strategy and market willingness to pay
• Distribution channels and go-to-market strategy
• Scalability and growth potential
• Risk factors and mitigation strategies""",
        help="Provide a detailed analysis based on your research and survey results"
    )
    
    st.session_state.marketfit = marketfit
    
    # Character count and validation
    char_count = len(marketfit)
    st.caption(f"Characters: {char_count} (Recommended: 200-1000)")
    
    # Generate Market Fit Feedback
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("📊 Generate Market Fit Analysis", type="primary", use_container_width=True):
            if not validate_api_key():
                return
                
            if not marketfit.strip() or char_count < 100:
                st.error("❌ Please provide a more detailed market fit analysis (at least 100 characters).")
            else:
                with st.spinner("🤖 Analyzing market fit..."):
                    try:
                        analysis_result = get_market_fit_feedback(marketfit)
                        st.session_state.market_fit_feedback = analysis_result
                        st.success("✅ Market fit analysis generated successfully!")
                        logger.info("Market fit feedback generated successfully")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error generating analysis: {str(e)}")
                        logger.error(f"Market fit analysis error: {str(e)}\n{traceback.format_exc()}")
    
    # Display Generated Feedback
    if st.session_state.get('market_fit_feedback'):
        st.markdown("---")
        st.subheader("💡 AI-Generated Market Fit Insights")
        
        with st.container():
            st.markdown(st.session_state.market_fit_feedback)
        
        # Evaluate Market Fit
        if st.button("🔍 Evaluate Market Fit Quality", type="secondary", use_container_width=True):
            with st.spinner("🔍 Evaluating market fit analysis..."):
                try:
                    evaluate_market_fit_with_openai = safe_import('evaluate_marketfit', 'evaluate_market_fit_with_openai')
                    if evaluate_market_fit_with_openai:
                        marketfit_text = st.session_state.get('marketfit', '')
                        if not marketfit_text.strip():
                            st.warning("⚠️ Please enter your market fit analysis before evaluation.")
                        else:
                            result = evaluate_market_fit_with_openai(marketfit_text)
                            st.session_state['market_fit_evaluation'] = result
                            st.success("✅ Market fit evaluation completed!")
                            logger.info("Market fit evaluation completed")
                    else:
                        st.error("❌ Market fit evaluation module not available")
                except Exception as e:
                    st.error(f"❌ Error evaluating market fit: {str(e)}")
                    logger.error(f"Market fit evaluation error: {str(e)}")
        
        # Show Evaluation Results
        if st.session_state.get('market_fit_evaluation'):
            with st.expander("📊 Market Fit Evaluation Results", expanded=True):
                evaluation_data = st.session_state['market_fit_evaluation']
                if isinstance(evaluation_data, dict):
                    for key, value in evaluation_data.items():
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                else:
                    st.json(evaluation_data)
    
    # Navigation
    st.markdown("---")
    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
    
    with nav_col1:
        if st.button("← Previous Step", use_container_width=True):
            navigate_to_page(4)
            st.rerun()
    
    with nav_col3:
        if st.session_state.get('market_fit_feedback'):
            if st.button("Next: Prototype →", type="primary", use_container_width=True):
                navigate_to_page(6)
                st.rerun()
        else:
            st.button("Next: Prototype →", disabled=True, use_container_width=True, 
                    help="Complete market fit analysis first")

# Page 6: Prototype Generation
def render_page_6():
    """Render prototype generation page"""
    st.title("🚀 Prototype Generator")
    st.header("Step 6: Generate Prototype Visualizations")

    if not st.session_state.get('selected_idea'):
        st.error("❌ Please complete previous steps before generating prototypes.")
        if st.button("🔙 Go Back to Start", use_container_width=True):
            navigate_to_page(1)
            st.rerun()
        return

    # Project Context Display
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**SDGs:** {', '.join(st.session_state.get('selected_sdgs', []))}")
        st.info(f"**Idea:** {st.session_state.selected_idea[:100]}...")
    
    with col2:
        st.info(f"**Problem:** {st.session_state.get('problem_statement', 'Not provided')[:100]}...")
    
    # Prototype Description Input
    st.subheader("📝 Prototype Description")
    prototype_description = st.text_area(
        "Describe your prototype visualization in detail:",
        value=st.session_state.get('prototype_description', ''),
        height=150,
        placeholder="""Describe your prototype including:
• Key features and functionality
• User interface design elements
• Materials and technology used
• How it solves the identified problem
• Target user experience
• Visual design preferences""",
        key="prototype_description_input",
        help="Be specific about visual elements, user interactions, and technical details"
    )
    
    st.session_state.prototype_description = prototype_description
    st.caption(f"Characters: {len(prototype_description)} (Recommended: 100-500)")
    
    # Style and Generation Options
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🎨 Visualization Style")
        
        style_options = {
            "Photorealistic Concept": "High-resolution, photorealistic visualization in real-world context",
            "3D Mockup": "Professional 3D render showcasing key features and materials",
            "Whiteboard Sketch": "Innovation sketch with annotations and business model elements",
            "User Interface (UI) Mockup": "Digital product mockup showing user journey and functionality",
            "Infographic Style": "Business-focused infographic with product visualization"
        }
        
        selected_style = st.selectbox(
            "Choose visualization style:",
            options=list(style_options.keys()),
            index=0,
            help="Select the style that best fits your prototype presentation needs"
        )
        
        st.info(f"**{selected_style}:** {style_options[selected_style]}")
    
    with col4:
        st.subheader("🔢 Generation Options")
        
        num_images = st.slider(
            "Number of images to generate:",
            min_value=1,
            max_value=4,
            value=2,
            help="Generate multiple variations of your prototype visualization"
        )
        
        st.metric("Images to Generate", num_images)
    
    # Validation
    can_generate = True
    validation_messages = []
    
    if not prototype_description or len(prototype_description.strip()) < 50:
        validation_messages.append("⚠️ Please provide a more detailed prototype description (at least 50 characters)")
        can_generate = False
    
    if not st.session_state.get('selected_idea'):
        validation_messages.append("⚠️ Please select an idea first")
        can_generate = False
    
    if not st.session_state.get('problem_statement'):
        validation_messages.append("⚠️ Please provide a problem statement first")
        can_generate = False
    
    # Advanced validation if available
    if can_generate:
        try:
            validate_student_input = safe_import('validate_student_description', 'validate_student_input')
            if validate_student_input:
                validation_result = validate_student_input(
                    st.session_state.selected_idea,
                    st.session_state.problem_statement,
                    prototype_description
                )
                
                if not validation_result.get('valid', False):
                    validation_messages.append(f"⚠️ Validation failed: {validation_result.get('message', 'Unknown validation error')}")
                    can_generate = False
                else:
                    st.success("✅ All validations passed! Ready to generate prototype images.")
                    
        except Exception as e:
            st.warning(f"⚠️ Validation module error: {str(e)}. Using basic validation.")
    
    # Show validation messages
    for message in validation_messages:
        st.warning(message)
    
    # Generation Button
    if st.button("🎨 Generate Prototype Images", type="primary", disabled=not can_generate, use_container_width=True):
        if not validate_api_key():
            return
            
        if can_generate:
            with st.spinner("🎨 Creating your prototype visualization... This may take 30-60 seconds."):
                try:
                    # Import prototype generation functions
                    generate_prototype_images = safe_import('protype_image_gen', 'generate_prototype_images')
                    StyleOption = safe_import('protype_image_gen', 'StyleOption')
                    
                    if not generate_prototype_images or not StyleOption:
                        st.error("❌ Prototype generation module not available. Please check 'protype_image_gen.py'")
                        return
                    
                    # Style mapping
                    style_mapping = {
                        "Photorealistic Concept": StyleOption.PHOTOREALISTIC,
                        "3D Mockup": StyleOption.MOCKUP_3D,
                        "Whiteboard Sketch": StyleOption.WHITEBOARD,
                        "User Interface (UI) Mockup": StyleOption.UI_MOCKUP,
                        "Infographic Style": StyleOption.INFOGRAPHIC
                    }
                    
                    selected_style_enum = style_mapping.get(selected_style, StyleOption.PHOTOREALISTIC)
                    api_key = get_openai_api_key()
                    
                    # Generate images
                    result = generate_prototype_images(
                        idea=st.session_state.selected_idea,
                        problem=st.session_state.problem_statement,
                        prototype_description=prototype_description,
                        style=selected_style_enum,
                        num_images=num_images,
                        api_key=api_key
                    )
                    
                    if result.get("success", False):
                        # Store results
                        st.session_state.prototype_images = result["images"]
                        st.session_state.prototype_prompt = result.get("prompt_used", "")
                        st.session_state.prototype_style = selected_style
                        st.session_state.prototype_generated = True
                        
                        st.success(f"✅ Successfully generated {result.get('num_generated', 0)} prototype image{'s' if result.get('num_generated', 0) > 1 else ''}!")
                        logger.info(f"Generated {result.get('num_generated', 0)} prototype images")
                        
                    else:
                        st.error(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
                        logger.error(f"Prototype generation failed: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"❌ An unexpected error occurred: {str(e)}")
                    logger.error(f"Prototype generation error: {str(e)}\n{traceback.format_exc()}")
            
            st.rerun()
    
    # Display Generated Images
    if st.session_state.get('prototype_generated') and st.session_state.get('prototype_images'):
        st.markdown("---")
        st.subheader("🎨 Generated Prototype Visualizations")
        
        # Generation info
        col5, col6 = st.columns(2)
        with col5:
            st.info(f"**Style:** {st.session_state.get('prototype_style', 'Unknown')}")
        with col6:
            st.info(f"**Images:** {len(st.session_state.prototype_images)} generated")
        
        # Display images
        images = st.session_state.prototype_images
        
        if len(images) == 1:
            # Single image display
            img_data = images[0]
            try:
                image_bytes = base64.b64decode(img_data["image_base64"])
                st.image(image_bytes, caption="Prototype Visualization", use_container_width=True)
                
                st.download_button(
                    label="📥 Download Image",
                    data=image_bytes,
                    file_name="prototype_visualization.png",
                    mime="image/png",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"❌ Error displaying image: {str(e)}")
                
        else:
            # Multiple images in grid
            cols = st.columns(min(len(images), 2))
            
            for idx, img_data in enumerate(images):
                try:
                    image_bytes = base64.b64decode(img_data["image_base64"])
                    
                    with cols[idx % 2]:
                        st.image(
                            image_bytes, 
                            caption=f"Variation {img_data.get('variation_number', idx + 1)}", 
                            use_container_width=True
                        )
                        
                        st.download_button(
                            label=f"📥 Download Variation {img_data.get('variation_number', idx + 1)}",
                            data=image_bytes,
                            file_name=f"prototype_variation_{img_data.get('variation_number', idx + 1)}.png",
                            mime="image/png",
                            key=f"download_btn_{idx}",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"❌ Error displaying image {idx + 1}: {str(e)}")
        
        # Prototype Evaluation Section
        st.markdown("---")
        st.subheader("📊 Prototype Evaluation")
        
        eval_col1, eval_col2 = st.columns(2)
        
        with eval_col1:
            if st.button("🔍 Evaluate Prototype Description", type="secondary", use_container_width=True):
                with st.spinner("🔍 Evaluating prototype description quality..."):
                    try:
                        create_balanced_classifier = safe_import('evaluate_ptototype_description', 'create_balanced_classifier')

                        if create_balanced_classifier:
                            # Get API key for evaluation
                            openai_api_key = os.getenv("OPENAI_API_KEY")
                            if not openai_api_key:
                                try:
                                    openai_api_key = st.secrets.get("OPENAI_API_KEY")
                                except Exception:
                                    openai_api_key = None

                            if not openai_api_key:
                                st.warning("⚠️ OpenAI API key not found. Using mock evaluation...")
                                st.session_state.prototype_evaluation = {
                                    "success": True,
                                    "x_axis_category": "HIGH",
                                    "y_axis_category": "TRIPLE",
                                    "processing_time": 0.1,
                                    "model_used": "mock-evaluation"
                                }
                            else:
                                # Initialize classifier with OpenAI key
                                classifier = create_balanced_classifier(api_key=openai_api_key)

                                idea_text = st.session_state.get('selected_idea', '')
                                prototype_desc = st.session_state.get('prototype_description', '')

                                result = classifier.classify_sync(idea_text, prototype_desc)
                                st.session_state.prototype_evaluation = {
                                    "success": getattr(result, 'success', True),
                                    "x_axis_category": getattr(result, 'x_axis_category', None),
                                    "y_axis_category": getattr(result, 'y_axis_category', None),
                                    "error": getattr(result, 'error', None),
                                    "processing_time": getattr(result, 'processing_time', None),
                                    "tokens_used": getattr(result, 'tokens_used', None),
                                    "model_used": getattr(result, 'model_used', None),
                                    "raw_response": getattr(result, 'raw_response', None),
                                }

                            st.success("✅ Prototype evaluation completed!")
                            logger.info("Prototype evaluation completed")
                        else:
                            st.error("❌ Evaluation module not available")

                    except Exception as e:
                        st.error(f"❌ Evaluation error: {str(e)}")
                        logger.error(f"Prototype evaluation error: {str(e)}")

        # Show evaluation results
        if st.session_state.get("prototype_evaluation"):
            with st.expander("📋 Prototype Evaluation Results", expanded=True):
                eval_data = st.session_state["prototype_evaluation"]
                if eval_data.get("success"):
                    col_eval1, col_eval2 = st.columns(2)
                    with col_eval1:
                        st.metric("Quality Assessment", eval_data.get('x_axis_category', 'N/A'))
                    with col_eval2:
                        st.metric("Content Assessment", eval_data.get('y_axis_category', 'N/A'))
                    
                    if eval_data.get('processing_time'):
                        st.write(f"**Processing Time:** {eval_data['processing_time']:.2f}s")
                    if eval_data.get('tokens_used'):
                        st.write(f"**Tokens Used:** {eval_data['tokens_used']}")
                    if eval_data.get('model_used'):
                        st.write(f"**Model:** {eval_data['model_used']}")
                else:
                    st.error(f"Evaluation failed: {eval_data.get('error', 'Unknown error')}")
        
        # Project Completion
        st.markdown("---")
        st.subheader("🎉 Project Completion")
        
        completion_col1, completion_col2, completion_col3 = st.columns(3)
        
        with completion_col2:
            if st.button("✅ Complete Project", type="primary", use_container_width=True):
                st.balloons()
                st.success("🎉 Congratulations! Your SDG Innovation project is complete!")
                
                # Project summary
                with st.expander("📋 Project Summary", expanded=True):
                    summary_data = {
                        "SDGs": ', '.join(st.session_state.get('selected_sdgs', [])),
                        "Idea": st.session_state.selected_idea[:200] + "..." if len(st.session_state.selected_idea) > 200 else st.session_state.selected_idea,
                        "Problem Statement": st.session_state.get('problem_statement', 'Not provided')[:200] + "..." if len(st.session_state.get('problem_statement', '')) > 200 else st.session_state.get('problem_statement', 'Not provided'),
                        "Target Market": st.session_state.get('target_market', 'Not defined'),
                        "Prototype Style": st.session_state.get('prototype_style', 'Not specified'),
                        "Images Generated": len(st.session_state.get('prototype_images', [])),
                        "Survey Questions": len(st.session_state.get('survey_questions', [])),
                        "Market Research": "✅ Completed" if st.session_state.get('market_research_results') else "❌ Not completed",
                        "Market Fit Analysis": "✅ Completed" if st.session_state.get('market_fit_feedback') else "❌ Not completed"
                    }
                    
                    for key, value in summary_data.items():
                        st.write(f"**{key}:** {value}")
                
                logger.info("Project completed successfully")
    
    # Navigation
    st.markdown("---")
    nav_col1, nav_col2 = st.columns([1, 1])
    
    with nav_col1:
        if st.button("← Previous Step", use_container_width=True):
            navigate_to_page(5)
            st.rerun()
    
    with nav_col2:
        if st.button("🔄 Start New Project", use_container_width=True):
            # Clear session state for new project
            keys_to_keep = ['page']
            keys_to_delete = [key for key in st.session_state.keys() if key not in keys_to_keep]
            for key in keys_to_delete:
                del st.session_state[key]
            navigate_to_page(1)
            st.rerun()

# Main application logic
def main():
    """Main application entry point"""
    try:
        # Render sidebar
        render_sidebar()
        
        # Route to appropriate page
        page_renderers = {
            1: render_page_1,
            2: render_page_2,
            3: render_page_3,
            4: render_page_4,
            5: render_page_5,
            6: render_page_6
        }
        
        current_page = st.session_state.get('page', 1)
        
        if current_page in page_renderers:
            page_renderers[current_page]()
        else:
            st.error(f"❌ Invalid page number: {current_page}")
            navigate_to_page(1)
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}\n{traceback.format_exc()}")
        
        if st.button("🔄 Restart Application"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# Footer
def render_footer():
    """Render application footer"""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 2rem 0;'>
            <p><strong>SDG Innovation Hub</strong> - Transforming Global Challenges into Innovation Opportunities</p>
            <p>Built with Streamlit • Powered by OpenAI GPT-4.1 • Version 2.0</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    render_footer()