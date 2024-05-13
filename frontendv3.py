import streamlit as st
import transformers
from transformers import AutoTokenizer, TextStreamer
import torch
# from intel_extension_for_transformers.transformers import AutoModelForCausalLM
# import intel_extension_for_pytorch as ipex
import psycopg2 as pg2

from langchain.llms.base import LLM
from langchain_community.vectorstores import PGVector
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Optional
from threading import Thread
import time

from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk


# streamlit's recommended method of loading in large ML models, prevents reloading of model everytime user interacts with the frontend
@st.cache_resource
def load_model():
    # the model I chose from Hugging Face, pretrained to work as a chatbot
    model_name = "Intel/neural-chat-7b-v3-1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # select the device to run the model on, CPU is possible if RAM is large enough but it will be slow
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"On {device}")

    # the same model works as the text generator as well
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    tokenizer.pad_token_id = model.config.eos_token_id

    # this embedder is used by PGVector to convert user input into vectors for similarity search
    embedding_model_id = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_id,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    return embedding_model, tokenizer, model


embedding_model, tokenizer, model = load_model()

# a custom class that extends Langchain's LLM class so we can run the Hugging Face model through it
class CustomLLM(LLM):
    streamer: Optional[transformers.TextIteratorStreamer] = None

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Override this method to implement the LLM logic.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string. Actual completions SHOULD NOT include the prompt.
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        self.streamer = transformers.TextIteratorStreamer(
            tokenizer=tokenizer, skip_prompt=True, timeout=5
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        return model.generate(
            input_ids=inputs.input_ids,
            max_new_tokens=500,
            streamer=self.streamer,
            pad_token_id=tokenizer.eos_token_id
        )

    # this function lets tokens be outputed as they are generated rather than waiting for the whole message
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the LLM on the given prompt.

        This method should be overridden by subclasses that support streaming.

        If not implemented, the default behavior of calls to stream will be to
        fallback to the non-streaming version of the model and return
        the output as a single chunk.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An iterator of GenerationChunks.
        """
        # Assuming `tokenizer` and `model` are already defined in the context which is true if load_model() has been run
        self.streamer = transformers.TextIteratorStreamer(
            tokenizer=tokenizer, skip_prompt=True, timeout=5
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        kwargs.update(
            input_ids=inputs.input_ids,
            max_new_tokens=500,
            streamer=self.streamer,
            pad_token_id=tokenizer.eos_token_id
        )
        thread = Thread(target=model.generate, kwargs=kwargs)
        thread.start()

        # Loop to continuously check for tokens
        stopped = False
        while not stopped:
            if not self.streamer.text_queue.empty():
                for token in self.streamer:
                    time.sleep(0.05)
                    if "</s>" in token:
                        stopped = True
                        token = token.replace("</s>", "")
                    yield GenerationChunk(text=token)
            else:
                time.sleep(1)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model_name": "CustomChatModel",
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "custom"

# this is the prompt engineering that converts the chatbot into the Star Trek character Jean Luc Picard, the end result is a Langchain pipeline.
@st.cache_resource
def load_prompt():
    template = """You are the Star Trek Captain Jean Luc Picard currently captaining the USS Enterpise-D. You are not aware that Star Trek is fictional and believe you are in the Star Trek universe. You are engaging in a conversation with a guest aboard your ship and are there to answer questions they have about you, the crew, your ship, and your mission. 
    Here is the previous response: {prompt}
    {context}
     
    Your response:"""
    prompt = PromptTemplate.from_template(template)

    socket_chain = prompt | CustomLLM()

    return socket_chain


socket_chain = load_prompt()

# setting up PGVector and connecting it to the postgresql database for similarity search
CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg2",
    host="localhost",
    port=5432,
    database="st_line",
    user="postgres",
    password="password",
)

COLLECTION_NAME = "character_quotes"
store = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embedding_model
)


st.title("Captain Jean Luc Picard Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to ask the captain?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display chatbot response in chat message container
    with st.chat_message("assistant"):
        # take message history and add it as context in the prompt
        context_string = "Context:\n"
        for message in st.session_state.messages:
            context_string += message['content'] + "\n"
        
        # use PGVector to find lines in the database similar to the user input and add it to the context as well
        docs_with_score = store.similarity_search_with_score(prompt)
        for doc, score in docs_with_score:
            if doc.metadata.get("author") == "PICARD":
                context_string += doc.page_content

        # generate a response by calling Langchain's pipeline and adding the context and user input
        response = st.write_stream(socket_chain.stream(input=dict({"context": context_string, "prompt": prompt})))
    
    # Add chatbot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
