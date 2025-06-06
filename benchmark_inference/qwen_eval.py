import dashscope
import time
import json


def qwen2_call(messages: list, model: str, token:str, streaming:bool=False):
    # dashscope.api_key = os.environ.get('DASHSCOPE_API_KEY')
    # dashscope.api_key = "sk-c13dc1dc01584bfba4c919022897f050"
    # dashscope.base_http_api_url = os.environ.get('BASE_HTTP_API_URL', 'https://dashscope.aliyuncs.com/api/v1')
    # print('call qwen2')
    # construct message

    retry_cnt = 0
    while retry_cnt < 2:
        try:
            if not streaming:
                response = dashscope.Generation.call(
                    model=model,
                    messages=messages,  # noqa
                    result_format='message',
                    stream=False,
                    top_p=0.95,
                    api_key = token,
                    temperature=1.0,
                    logprobs=True, 
                    top_logprobs=5,
                    headers={'X-DashScope-DataInspection': 'disable'},
                )

                choices = [dict(choices) for choices in response['output']["choices"]]
                for i, choice in enumerate(choices):
                    choices[i]["text"] = choice["message"]["content"]

                # choices_return = [dict(text=choice["message"]["content"], total_tokens=0) for i, choice in enumerate(choices)]
                if choices is None:
                    logging.warning(f"Max retries reached. Returning empty completions.")
                    # TODO: the return is a tmp hack, should be handled better
                    choices = [dict(text="", total_tokens=0)] * len(prompt_batch)
               
                for choice in choices:
                    choice["total_tokens"] = response["usage"]["total_tokens"] / 1
                print(choices)

                return response['output']['choices'][0]['message']['content']
            else:
                for response in dashscope.Generation.call(
                    model=model,
                    messages=messages,
                    result_format='message',
                    api_key= token,
                    top_p=0.95,
                    stream=True,
                    headers={'X-DashScope-DataInspection': 'disable'},
                ):
                    if response['output']['finish_reason'] in ['stop', 'length']:
                        return response['output']['choices'][0]['message']['content']
        except:
            retry_cnt += 1

            time.sleep(2)
            print("FAILED!",response)
            continue
    return ""

if __name__ == "__main__":
    api_key = "sk-c13dc1dc01584bfba4c919022897f050"
    qwen2_call(messages=[{'role':'user', 'content':"tell me your name"}],  model='qwen-max', token=api_key)