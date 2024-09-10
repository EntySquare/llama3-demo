import fire
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from llama import Llama, Dialog

app = FastAPI()


# Initialize the LLaMA model
def get_generator(ckpt_dir: str, tokenizer_path: str, max_seq_len: int, max_batch_size: int):
    return Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )


class ChatRequest(BaseModel):
    input_text: str
    temperature: float = 0.6
    top_p: float = 0.9
    max_gen_len: Optional[int] = None


def main(
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 512,
        max_batch_size: int = 4,
        max_gen_len: Optional[int] = None
):
    global generator
    generator = get_generator(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


@app.post("/generate/")
def generate_response(request: ChatRequest):
    dialogs: List[Dialog] = [
        [{"role": "user", "content": request.input_text}]
    ]
    results = generator.chat_completion(
        dialogs,
        max_gen_len=request.max_gen_len,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    response = results[0]['generation']['content']
    return {"response": response}


if __name__ == "__main__":
    fire.Fire(main)
