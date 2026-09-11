from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def create_rag_chain():
    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are AtomBot, a helpful assistant that answers questions
        using the provided document.

        Use the document context and conversation history to answer
        the user's question.

        If the answer is not available in the document, say:
        "I could not find this information in the document."

        Do not make up information.

        Conversation history:
        {history}

        Document context:
        {context}

        Current question:
        {question}

        Answer:
        """
    )

    def ask(question, history=""):
        documents = retriever.invoke(question)

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        response = llm.invoke(
            prompt.format(
                history=history,
                context=context,
                question=question
            )
        )

        return response.content, documents

    return ask