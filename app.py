import streamlit as st
from streamlit_ace import st_ace
from langgraph.graph import StateGraph
import logging
import re
import ast
from typing import TypedDict
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(model="qwen-2.5-32b")

# Define State Schema
class CodeReviewState(TypedDict):
    language: str
    code: str
    review: str
    error: str

def validate_response_format(response: str) -> bool:
    """Validate LLM response structure"""
    required_sections = [
        r"^\*\*Issues:\*\*",
        r"^\*\*Corrected Code \(Preserve Indentation\):\*\*",
        r"^\*\*Optimized Code:\*\*",
        r"^\*\*Explanation:\*\*"
    ]
    return all(re.search(section, response, re.MULTILINE) for section in required_sections)

def validate_python_syntax(code: str) -> str:
    """Validate Python syntax using AST"""
    try:
        ast.parse(code)
        return ""
    except SyntaxError as e:
        return f"🔴 Syntax Error in Generated Code:\n{str(e)}"

def extract_code_blocks(response: str, language: str) -> tuple:
    """Extract code blocks from LLM response"""
    corrected_pattern = re.compile(
        fr"```{language}\n(.*?)```", 
        re.DOTALL
    )
    optimized_pattern = re.compile(
        fr"```{language}\n(.*?)```", 
        re.DOTALL
    )
    
    corrected_match = corrected_pattern.search(response)
    optimized_match = optimized_pattern.search(response)
    
    return (
        corrected_match.group(1).strip() if corrected_match else None,
        optimized_match.group(1).strip() if optimized_match else None
    )

def review_code(state: CodeReviewState) -> CodeReviewState:
    """Generate code review and corrections"""
    language = state["language"]
    original_code = state["code"].strip()
    
    prompt = f"""### Code Review Task for {language}
Analyze this code for errors, correct them while preserving indentation, 
then create an optimized version. Follow this format:

**Issues:** List problems found
**Corrected Code (Preserve Indentation):** {language} code block with fixes
**Optimized Code:** {language} code block with improvements
**Explanation:** Benefits of optimizations

Original Code:
```{language}
{original_code}
```"""

    try:
        response = llm.invoke(prompt).content
    except Exception as e:
        return {"error": f"API Error: {str(e)}"}
    
    if not validate_response_format(response):
        return {"error": "Invalid response format from AI model"}
    
    validation_msg = ""
    if language.lower() == "python":
        corrected_code, _ = extract_code_blocks(response, language)
        if corrected_code:
            validation_msg = validate_python_syntax(corrected_code)
    
    if validation_msg:
        response += f"\n\n{validation_msg}"
    
    return {
        "language": language,
        "code": original_code,
        "review": response,
        "error": ""
    }

# Streamlit UI
st.title("🧑💻 AI Code Review Assistant")

# Language selection
language = st.selectbox(
    "Select Programming Language:",
    ["Python", "JavaScript", "Java", "C++", "Go"],
    index=0
)

# Code editor
code_input = st_ace(
    language=language.lower(),
    theme="monokai",
    height=300,
    wrap=True,
    font_size=14,
    key="code_editor"
)

# Review button
if st.button("🔍 Analyze Code"):
    if not code_input.strip():
        st.warning("Please enter code to analyze")
    else:
        with st.spinner("Analyzing code..."):
            state = {
                "language": language,
                "code": code_input,
                "review": "",
                "error": ""
            }
            result = review_code(state)
        
        if result["error"]:
            st.error(f"Error: {result['error']}")
        else:
            st.subheader("Analysis Results")
            st.markdown(result["review"], unsafe_allow_html=True)

# LangGraph workflow setup
workflow = StateGraph(CodeReviewState)
workflow.add_node("code_review", review_code)
workflow.set_entry_point("code_review")
workflow.set_finish_point("code_review")

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Application startup completed")