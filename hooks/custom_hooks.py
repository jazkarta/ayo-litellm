import litellm
import json
import os
import httpx
from litellm.integrations.custom_logger import CustomLogger
from litellm import completion, acompletion


class MyCustomHandler(CustomLogger):

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Async Success")
        
        user_raw = kwargs.get('user')
        user_id = user_raw
        user_email = None
        conversation_id = None
        message_id = None
        
        if user_raw and isinstance(user_raw, str) and user_raw.startswith('{'):
            try:
                user_data = json.loads(user_raw)
                user_id = user_data.get('id')
                user_email = user_data.get('email')
                conversation_id = user_data.get('conversationId')
            except Exception:
                pass

        litellm_params = kwargs.get('litellm_params', {})
        metadata = litellm_params.get('metadata', {})
        
        messages = kwargs.get('messages', [])
        prompt = messages
        last_message = messages[-1].get('content') if messages else None
        
        ai_response = None
        if hasattr(response_obj, 'choices') and len(response_obj.choices) > 0:
            ai_response = response_obj.choices[0].message.content
            
        model_name = getattr(response_obj, 'model', None)

        payload = {
            'user_email': user_email,
            'conversation_id': conversation_id,
            'model_name': model_name,
            'prompt': last_message,
            'response': ai_response,
        }

        api_base_url = os.getenv("API_BASE_URL")

        print(f"API BASE URL {api_base_url}")
        if api_base_url:
            api_url = f"{api_base_url.rstrip('/')}/api/chats/"
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(api_url, json=payload)
                    response.raise_for_status()
                    print(f"Successfully sent payload to {api_url}")
            except Exception as e:
                print(f"Error calling API {api_url}: {e}")
        else:
            print("API_BASE_URL not set in environment variables.")

customHandler = MyCustomHandler()