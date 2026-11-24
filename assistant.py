# standard library imports
import os

# third-party imports
from openai import OpenAI
from dotenv import load_dotenv
from types import SimpleNamespace

# local module imports
from tool import TOOLS,run_tool
from utilities.utilities import log


load_dotenv()


client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
);

def stream_chat(history):
    stream = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=history,
        tools=TOOLS,
        stream=True,
    )

    full_response = ""
    calls = {}
    label_printed = False

    # streaming started
    try:
        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                if not label_printed:
                    print("Assistant: ",end="",flush=True)
                    label_printed = True
                print(delta.content,end="" ,flush=True)
                full_response += delta.content

            # 2. Tools Calls
            for tc in (delta.tool_calls or []):
                obj = calls.setdefault(
                    tc.index,
                    SimpleNamespace(id=None,function=SimpleNamespace(name="",arguments=""))
                )
                
                if tc.id:
                    obj.id = tc.id
                if tc.function and tc.function.name:
                    obj.function.name += tc.function.name
                if tc.function and tc.function.arguments:
                    obj.function.name +=tc.function.name
               
    except KeyboardInterrupt:
        print("\n{Stopped by you}")
        calls = {}

    
    if label_printed:
        print() # move to next line after the assistant response 

    calls_list = []

    for key in sorted(calls):
        calls_list.append(calls[key])   

    return full_response,calls_list    



def build_tool_calls_payload(tool_calls):
    # converts over collected tools calls into dict format

    payload = []
    for tool_call in tool_calls:
        tool_calls_dict = {
            "id":tool_call.id,
            "type":"function",
            "function":{
                "name":tool_call.function.name,
                "arguments":tool_call.function.arguments
            }
        }
        payload.append(tool_calls_dict)

    return payload

message = []

message.append(
    {
        "role":"system",
        "content" : "You are a funny person and always give answer within 50 words"
    }
)

MAX_STEPS = 5

while True:

    user_input=input("\nYou:")

    if user_input.lower() in ["exit","bye","thik","goodbye"]:
        break

    message.append({
        "role":"user",
        "content":user_input
    })

    for step in range(MAX_STEPS):

        full_response,tool_calls = stream_chat(message)
            

        if not tool_calls:
            break

        log(f"Step {step + 1}: model requested {len(tool_calls)} tool calls")
        
        message.append({
            "role":"assistant",
            "content":full_response,
            "tool_calls":build_tool_calls_payload(tool_calls)
        })

        for tool_call in tool_calls:
            result = run_tool(tool_call)
            message.append({
                "role":"tool",
                "tool_call_id":tool_call.id,
                "content": str(result)
            })

    # else loop run when for loop run completely without breaking the loop
    else: 
        # Ran out of steps:THe model never gave the result
        fallback = "Sorry, I could not finish this within the steps limit."
        message.append(
                {
                    "role":"assistant",
                    "content":fallback
                }
            )
        print("Assistant: "+ fallback)
        continue


        























    

