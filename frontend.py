import streamlit as st
import transformers
from transformers import AutoTokenizer, TextStreamer
import torch
# from intel_extension_for_transformers.transformers import AutoModelForCausalLM
# import intel_extension_for_pytorch as ipex
import psycopg2 as pg2

from langchain.llms.base import LLM
from langchain_community.vectorstores.pgvector import PGVector
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Optional
from threading import Thread
import time
import asyncio

model_name = "Intel/neural-chat-7b-v3-1"
prompt = "Once upon a time, there existed a little girl,"

@st.cache_resource
def load_model():
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

  # streamer = TextStreamer(tokenizer)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"On {device}")

  model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
  tokenizer.pad_token_id = model.config.eos_token_id

  return tokenizer, model

tokenizer, model = load_model()

class CustomSocketLLM(LLM):
    streamer: Optional[TextStreamer] = None
    history = []
    session_saved = False

    def _call(self, prompt, stop=None, run_manager=None) -> str:
        self.streamer = TextStreamer(tokenizer)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        kwargs = dict(
            input_ids=inputs.input_ids,
            max_new_tokens=50,
            streamer=self.streamer,
            pad_token_id=tokenizer.eos_token_id,
        )
        thread = Thread(target=model.generate, kwargs=kwargs)
        thread.start()
        return ""

    @property
    def _llm_type(self) -> str:
        return "custom"

    def stream_tokens(self):
        print("stream working")
        if not self.streamer.text_queue.empty():
            print(self.streamer.text_queue)
            for token in self.streamer:
                time.sleep(0.05)
                self.history[-1] += token
                print(token)
                yield(token)
                
        else:
            print("empty")
            time.sleep(1)
            self.stream_tokens()

@st.cache_resource
def load_prompt():
  template = """You are the a chatbot. You are engaging in a conversation with a human.
  Here is the previous response: {prompt}
  Always respond politely and engage the opponent with clear, understandable language.
  Your response:"""
  prompt = PromptTemplate.from_template(template)

  socket_llm = CustomSocketLLM()
  socket_chain = prompt | socket_llm

  return socket_llm, socket_chain

socket_llm, socket_chain = load_prompt()

st.title("Echo Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # inputs = tokenizer(prompt, return_tensors="pt").input_ids
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    print(socket_chain)
    # socket_chain.invoke(input=dict({"prompt": prompt}))
    print(socket_llm)
    # output = socket_llm.stream_tokens()
    print("output")

    # response = st.write_stream(socket_llm.stream_tokens())
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
      socket_chain.invoke(input=dict({"prompt": prompt}))
      time.sleep(60)
      response = st.write_stream(socket_llm.stream_tokens())
      print("output 2")
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})