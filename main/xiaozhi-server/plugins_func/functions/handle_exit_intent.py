from plugins_func.register import register_function, ToolType, ActionResponse, Action
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

handle_exit_intent_function_desc = {
                "type": "function",
                "function": {
                    "name": "handle_exit_intent",
                    "description": "Called when the user wants to end the conversation or exit the system",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "say_goodbye": {
                                "type": "string",
                                "description": "A friendly farewell message to end the conversation with the user"
                            }
                        },
                        "required": ["say_goodbye"]
                    }
                }
            }

@register_function('handle_exit_intent', handle_exit_intent_function_desc, ToolType.SYSTEM_CTL)
def handle_exit_intent(conn, say_goodbye: str):
    # Handle exit intent
    try:
        conn.close_after_chat = True
        logger.bind(tag=TAG).info(f"Exit intent handled: {say_goodbye}")
        return ActionResponse(action=Action.RESPONSE, result="Exit intent handled", response=say_goodbye)
    except Exception as e:
        logger.bind(tag=TAG).error(f"Error handling exit intent: {e}")
        return ActionResponse(action=Action.NONE, result="Failed to handle exit intent", response="")