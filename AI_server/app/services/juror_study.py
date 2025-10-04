from transformers import pipeline
from app.models.schemas import STTRequest, RequestInput, ResponseOutput, LastInput, LastOutput, AttackdefenseInput
import os
import httpx
from dotenv import load_dotenv
load_dotenv()

