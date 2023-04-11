from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from translate import *


class CustomParam:
    def __init__(self):
        self.src_lang = ""
        self.tgt_lang = ""
        self.model_path = ""
        self.BPE_path = "./data/BPE_with_comments_codes"
        self.beam_size = 1

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/translation")
async def translation(language1: str, language2: str, code1: str):
    if language1 == "1":
        # Cpp
        f = open("./codes/sample.cpp", "w")
        f.write(code1)
        f.close()
    elif language2 == "2":
        # Java
        f = open("./codes/sample.java", "w")
        f.write(code1)
        f.close()
    else:
        # Python
        f = open("./codes/sample.py", "w")
        f.write(code1)
        f.close()

    params = CustomParam()
    cond1 = (language1=="1" and language2=="2")
    cond2 = (language1=="2" and language2=="1")
    cond3 = (language1=="2" and language2=="3")
    cond4 = (language1=="1" and language2=="3")
    cond5 = (language1=="3" and language2=="1")
    cond6 = (language1=="3" and language2=="2")
    if cond1:
        params.src_lang = "cpp"
        params.tgt_lang = "java"
        params.model_path = "./models/model_1.pth"
    elif cond2:
        params.src_lang = "java"
        params.tgt_lang = "cpp"
        params.model_path = "./models/model_1.pth"
    elif cond3:
        params.src_lang = "java"
        params.tgt_lang = "python"
        params.model_path = "./models/model_1.pth"
    elif cond4:
        params.src_lang = "cpp"
        params.tgt_lang = "python"
        params.model_path = "./models/model_2.pth"
    elif cond5:
        params.src_lang = "python"
        params.tgt_lang = "cpp"
        params.model_path = "./models/model_2.pth"
    elif cond6:
        params.src_lang = "python"
        params.tgt_lang = "java"
        params.model_path = "./models/model_2.pth"
   
    # Initialize translator
    translator = Translator(params)
    # input = code1
    src_file = None
    input = None
    if language1 == "1":
        src_file = "./codes/sample.cpp"
    elif language1 == "2":
        src_file = "./codes/sample.java"
    elif language1 == "3":
        src_file = "./codes/sample.py"
    with open(src_file, "r") as src:
        input = src.read()

    with torch.no_grad():
        output = translator.translate(
            input, lang1=params.src_lang, lang2=params.tgt_lang, beam_size=params.beam_size)

    code2 = output[0]
    
    return {"code2": code2}