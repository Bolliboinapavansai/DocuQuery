from flask import Flask, render_template, request, jsonify, session
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.agents import AgentType, load_tools, initialize_agent
#from langchain_openai import OpenAI  # Updated import
from openai import OpenAI
import json

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z'  # Change this to a secure random key in production
serpapi_key = "aac6a5d560123d85a707d259994b561bc195d898e35876f4c4dc1755703c6742"

DB_FAISS_PATH = 'vectorstore/db_faiss'


custom_prompt_template = """
Hey there! I have a question for you. If you know the exact answer,
please provide it. However, if you're uncertain or don't have the information,
 simply respond with 'I don't know.' Ready? Here it is:

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


def retrieval_qa_chain(llm, prompt, retriever):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=retriever,
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain


def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm


def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={'k': 2})
    return llm, qa_prompt, retriever


def final_result(query, conversation_history):
    llm, qa_prompt, retriever = qa_bot()
    print(query)
    qa_chain = retrieval_qa_chain(llm, qa_prompt, retriever)
    try:
        response = qa_chain(query)
        result_text = response["result"]
        print(result_text)
        # Check if the response contains "I don't know"
        if "I don't know" in result_text:
            print('yes')
            # If it does, ask OpenAI for a response
            # Construct the message as a list of dictionaries
            client = OpenAI(api_key="sk-proj-f4PuDHjQYX4nCgKesn6dT3BlbkFJVHlGjE3734iFDmHM7e0B")
            message = [
                {
                    "role": "user",
                    "content": query
                }
            ]

            # Convert the message to a JSON string (if needed)
            message_json = json.dumps(message)
            print(message_json)

            # Generate chat completion
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=message,
                max_tokens=150,
            )

            # Get the completed text
            result_text = response.choices[0].message.content
            print(result_text)
        conversation_history.append({'text': query, 'author': 'user'})
        conversation_history.append({'text': result_text, 'author': 'bot'})
        return result_text, conversation_history
    except Exception as e:
        print(e)
        error_message = "I encountered an error while processing your request. Please try again later."
        conversation_history.append({'text': query, 'author': 'user'})
        conversation_history.append({'text': error_message, 'author': 'bot'})
        return error_message, conversation_history


@app.route('/')
def index():
    session.clear()  # Clear session data when accessing the index page
    return render_template('index.html')


@app.route('/conversation', methods=['POST'])
def conversation():
    conversation_history = session.get('conversation', [])
    query = request.json['conversation'][-1]['text']
    response, conversation_history = final_result(query, conversation_history)
    session['conversation'] = conversation_history
    return jsonify(conversation=conversation_history)


if __name__ == '__main__':
    app.run(debug=True)
