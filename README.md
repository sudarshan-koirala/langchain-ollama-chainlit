# langchain-Ollama-Chainlit

Simple Chat UI as well as chat with documents using LLMs with Ollama (mistral model) locally, LangChaiin and Chainlit
  
In these examples, we’re going to build a simpel chat UI and a chatbot QA app. We’ll learn how to:
#### Example 1
- Create a simple Chat UI locally.

#### Example 2
- Ingest documents into vector database, store locally (creates a knowledge base)
- Create a chainlit app based on that knowledge base.

#### Example 3
- Upload a document(pdf)
- Create vector embeddings from that pdf
- Create a chatbot app with the ability to display sources used to generate an answer
---

### Chat with your documents 🚀
- [Ollama](https://ollama.ai/) and `mistral`as Large Language model
- [LangChain](https://python.langchain.com/en/latest/modules/models/llms/integrations/huggingface_hub.html) as a Framework for LLM
- [Chainlit](https://docs.chainlit.io/) for deploying.

## System Requirements

You must have Python 3.9 or later installed. Earlier versions of python may not compile.  

---

## Steps to Replicate 

1. Fork this repository and create a codespace in GitHub as I showed you in the youtube video OR Clone it locally.
```
git clone https://github.com/sudarshan-koirala/langchain-ollama-chainlit.git
cd langchain-ollama-chainlit
```

2. Rename example.env to .env with `cp example.env .env`and input the langsmith environment variables. This is optional.

3. Create a virtualenv and activate it
   ```
   python3 -m venv .venv && source .venv/bin/activate
   ```

4. Run the following command in the terminal to install necessary python packages:
   ```
   pip install -r requirements.txt
   ```

5. Run the following command in your terminal to start the chat UI:
   ```
   # Example 1
   chainlit run simple_chatui.py
   ```
   ---
    ```
   # Example 2
   python3 ingest.py
   chainlit run main.py
   ```
   ---
    ```
   # Example 3
   chainlit run rag.py
   ```

   ---
## Disclaimer
This is test project and is presented in my youtube video to learn new stuffs using the openly available resources (models, libraries, framework,etc). It is not meant to be used in production as it's not production ready. You can modify the code and use for your usecases !!
