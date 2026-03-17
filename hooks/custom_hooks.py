import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm import completion, acompletion


class MyCustomHandler(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs): 
        print(f"Pre-API Call")
    
    def log_post_api_call(self, kwargs, response_obj, start_time, end_time): 
        print(f"Post-API Call")

    def log_success_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Success")

    def log_failure_event(self, kwargs, response_obj, start_time, end_time): 
        print(f"On Failure")
    
    #### ASYNC #### - for acompletion/aembeddings

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Async Success")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        print(f"On Async Failure")

customHandler = MyCustomHandler()