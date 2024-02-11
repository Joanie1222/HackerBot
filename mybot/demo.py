#!/usr/bin/python3
from flask import Flask, render_template, request
import random
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Determining whether a GPU (CUDA-enabled) is available on your system
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# Function to handle conversations and generate responses
def generate_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

@app.route("/")
def home():
    return render_template("index.html", botName="Demo ChatBot", botAvatar="/static/hacker.jpg")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    botReply = generate_response(userText)
    noResponse = ["I don't know.", "I'm not sure about that.", "Is there a different way you can ask that?", "I don't have a response for that.", "I will have to give that some thought.", "I don't really know what you are asking."]
    if not botReply:
        botReply = random.choice(noResponse)
    return botReply

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
