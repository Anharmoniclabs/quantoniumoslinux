"""Optional llama.cpp constrained tool decoding; NESSA still validates all calls."""
from agentharness.llm import ChatClient, ModelError
from agentharness.tools import build_tools

def tool_response_schema(offered):
    registry=build_tools(allow_shell=False,allow_extract=True)
    variants=[]
    for name in sorted(offered):
        if name not in registry:raise ValueError('Unknown offered tool: '+name)
        params=registry[name].schema()['function']['parameters']
        params['additionalProperties']=False
        variants.append({'type':'object','properties':{'name':{'const':name},'arguments':params},
                         'required':['name','arguments'],'additionalProperties':False})
    if not variants:raise ValueError('No tools offered')
    return {'oneOf':variants}

class ConstrainedLocalClient(ChatClient):
    def chat(self,messages,tools=None,tool_names=None):
        if tools:raise ModelError('Constrained local adapter expects the text-tool protocol')
        self.response_schema=tool_response_schema(tool_names or set())
        return super().chat(messages,tools,tool_names)
    def _request(self,route,body=None,timeout=None):
        if route=='/chat/completions' and body is not None:
            body=dict(body)
            body['response_format']={'type':'json_object','schema':self.response_schema}
            body['chat_template_kwargs']={'enable_thinking':False}
        return super()._request(route,body,timeout)
