from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from utils.config import LLM_MODEL, OPENAI_API_KEY


def get_pipeline(retrieved_docs, llm=None):
    llm = llm or ChatOpenAI(api_key=OPENAI_API_KEY, model=LLM_MODEL, temperature=0.7)
    
    # Initialize empty history within the closure
    history = []
    
    prompt = ChatPromptTemplate.from_template(
    """
    ## System Instructions

    You are a specialized web scraping assistant designed to process and respond to user queries based solely on Context data. Your purpose is to deliver precise, factual information extracted from websites without deviation you must rely on data available through context and chat history only.

    Context:
    {context}

    Chat History:
    {chat_history}

    Question: {question}

    Answer:
    Generate a response strictly based on the provided context and chat history. The response should be:

    Accurate and Relevant: Only use the information available in the context and chat history. Do not include assumptions, speculations, or external knowledge.

    Concise and Informative: Deliver the answer in a clear, structured, and professional manner. Avoid unnecessary elaboration.
    
    Objective and Context-Aware: If the requested information is not present in the context or chat history, explicitly state that the required details are not available instead of guessing or providing unrelated information.
    
    No Small Talk or Off-Topic Responses: The response should be strictly business-oriented and focused on addressing the user's query based on the given data. Also, if encountered any small talk of off talks then just ask them to stick to the topic.
    
    This approach ensures the chatbot maintains reliability, consistency, and industry-grade accuracy.
    """)

    def format_docs(docs):
        return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

    def format_history():
        formatted = ""
        for i in range(0, len(history), 2):
            if i < len(history):
                formatted += f"Human: {history[i]}\n"
            if i+1 < len(history):
                formatted += f"AI: {history[i+1]}\n"
        return formatted

    # Build the core chain
    chain = (
        {
            "context": lambda _: format_docs(retrieved_docs),
            "chat_history": lambda _: format_history(),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Create a pipeline function that handles the query and updates history
    def pipeline(query, retrieved_docs):
        context = format_docs(retrieved_docs)
        print(f"context updated with {len(retrieved_docs)} new documents")
        chat_history = format_history()
        response = chain.invoke({"context": context, "chat_history": chat_history, "question": query})
        history.append(query)
        history.append(response)
        return response


    # Method to clear history
    def clear_history():
        history.clear()

    # Add clear_history as an attribute of the pipeline function
    pipeline.clear_history = clear_history
    return pipeline