import yaml
from path_tool import get_abs_path

def load_rag_config(path:str = get_abs_path("config/rag.yaml"),encoding:str="utf-8"):
    with open(path,"r",encoding=encoding) as f:
        rag_config = yaml.load(f, Loader=yaml.FullLoader)
        return rag_config
def load_chroma_config(path:str = get_abs_path("config/chroma.yaml"),encoding:str="utf-8"):
    with open(path,"r",encoding=encoding) as f:
        rag_config = yaml.load(f, Loader=yaml.FullLoader)
        return rag_config
def load_prompts_config(path:str = get_abs_path("config/prompts.yaml"),encoding:str="utf-8"):
    with open(path,"r",encoding=encoding) as f:
        rag_config = yaml.load(f, Loader=yaml.FullLoader)
        return rag_config
def load_agent_config(path:str = get_abs_path("config/agent.yaml"),encoding:str="utf-8"):
    with open(path,"r",encoding=encoding) as f:
        rag_config = yaml.load(f, Loader=yaml.FullLoader)
        return rag_config