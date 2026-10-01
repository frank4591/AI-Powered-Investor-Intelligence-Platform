import os
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.messages import SystemMessage, HumanMessage
from .model_selector import get_next_gemini_model_name

load_dotenv()

def get_gemini_model(model : str | None = None)-> ChatGoogleGenerativeAI:
    model_name = model or get_next_gemini_model_name()
    key = os.getenv("GOOGLE_GEMINI_KEY")
    if not model_name:
        raise ValueError("set google_gemini_model")
    if not key:
        raise ValueError("key not found")

    print(f"model selected {model_name}")
    return ChatGoogleGenerativeAI(
            model=model_name,
            api_key=key,
            temperature=0,
        )

def query_model(prompt :str) -> str :

    try:
        llm = get_gemini_model()
        prompt_template = ChatPromptTemplate([HumanMessage(content=prompt)])
        prompt = prompt_template.format()
        print(f"prompt: {prompt}")
        response = llm.invoke(prompt)
        content=  response.content
        print(f"llm response:{content}")
        if isinstance(content,str):
            return content
        if isinstance(content,list):
            text_part= [block["text"]
                        for block in content
                        if isinstance(block,dict)
                        and block.get("type") == "text"
                        and isinstance(block.get("text"),str)]
            if text_part:
                return "\n".join(text_part)
        return content
            
    except Exception as e:
        raise RuntimeError("error while quering llm" , e)
    raise
    

def get_structured_completion(
    prompt: str,
    response_model: type[BaseModel],
    model: str | None = None
) -> BaseModel:
    """
    Generate structured output.

    Args:
        prompt: Input prompt.
        response_model: Pydantic response model.
        model: Azure OpenAI deployment name.

    Returns:
        Parsed response model.
    """

    llm = get_gemini_model()

    structured_llm = llm.with_structured_output(response_model)
    response = structured_llm.invoke(
        [
            ("system", "You are an expert financial analyst."),
            ("human", prompt),
        ]
    )
    return response


def main():            
    # Build chat prompt – include retrieved context and the user question
    prompt = f"what is computer , describe in short"
    query_model(prompt)


if __name__ == "__main__":
     main()
    