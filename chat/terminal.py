import fire
from typing import List, Optional
from llama import Llama, Dialog


def generate_response(generator, input_text: str, temperature: float = 0.6, top_p: float = 0.9,
                      max_gen_len: Optional[int] = None) -> str:
    dialogs: List[Dialog] = [
        [{"role": "user", "content": input_text}]
    ]
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    response = results[0]['generation']['content']
    return response


def main(
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 512,
        max_batch_size: int = 4,
        max_gen_len: Optional[int] = None
):
    """
    Example usage of the LLaMA 3 model. Prompts correspond to chat turns between the user and assistant.

    `max_seq_len` needs to be <= 8192 for LLaMA models.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    print("欢迎使用 XAI 模型！输入你的问题，然后按回车键获取回复。输入 'exit' 退出程序。")

    while True:
        user_input = input("\n你: ")

        if user_input.lower() == "exit":
            print("退出程序...")
            break

        response = generate_response(generator, user_input, temperature=temperature, top_p=top_p,
                                     max_gen_len=max_gen_len)
        print("\nXAI: " + response)


if __name__ == "__main__":
    fire.Fire(main)
