import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

messages = []

messages.append({
    "role": "system",
    "content": "You are a helpful assistant.Give answer in under 50 words."
})


while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    messages.append({
        "role": "user",
        "content": user_input
    })

    stream = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=messages,
        stream=True
    )

    full_response = ""
    for chunk in stream:
        piece = chunk.choices[0].delta.content
        if piece:
            print(piece,end="",flush=True)
            full_response +=piece

    messages.append({
        "role": "assistant",
        "content": full_response
    })


