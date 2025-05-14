# 프롬프트 생성 -> LLM처리 -> 응단 반환
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
llm.invoke("지구의 자전 주기는?")