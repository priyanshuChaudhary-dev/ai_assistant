from datetime import datetime
from utilities.utilities import log
import json
import requests

def getCurrentTime():
    return datetime.now().isoformat()

def getCityWeather(city):
    url = f"https://wttr.in/{city}?format=3"
    response = requests.get(url,timeout=10)
    response.raise_for_status() # this will raise error when api error
    return response.text

def getUserCity():
    return "Meerut"

TOOLS = [
    {
        "type":"function",
        "function":{
            "name":"getCurrentTime",
            "description":"Returns the current date and time",
        }
    },
    {
        "type":"function",
        "function":{
            "name":"getCityWeather",
            "description":"Return the current weather of the given city",
            "parameters":{
                "type":"object",
                "properties":{
                    "city":{
                        "type":"string",
                        "description":"City to get the weather for"
                    }
                },
                "required":["city"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"getUserCity",
            "description":"Return the user city where the user live,'Home of user',current location of user"
        }
    }
]

FUNCTION_NAME_TO_FUNCTION_MAP = {
    "getCurrentTime":getCurrentTime,
    "getCityWeather":getCityWeather,
    "getUserCity":getUserCity
}


def run_tool(tool_call):
    func_name = tool_call.function.name
    func_args = {}
    try:
        func_args = json.loads(tool_call.function.arguments or "{}")
        log(f"🛠️  {func_name}({func_args})")
        fn = FUNCTION_NAME_TO_FUNCTION_MAP[func_name]
        result = fn(**func_args)
    except Exception as e:
        result = f"Error while running {func_name} : {e}"
        
        log(f"    {result}")
        return str(result);
                