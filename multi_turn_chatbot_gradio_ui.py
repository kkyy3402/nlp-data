from transformers import AutoTokenizer, AutoModelForCausalLM

# model, tokenizer 선언
tokenizer = AutoTokenizer.from_pretrained("heegyu/polyglot-ko-5.8b-chat")
model = AutoModelForCausalLM.from_pretrained("heegyu/polyglot-ko-5.8b-chat").to('cuda:0')

# multiturn 데이터 구성
def get_answer_from_chatbot(instruction):
    input_ids = tokenizer.encode(instruction, return_tensors='pt').to('cuda:0')
    gened = model.generate(
        input_ids,
        max_new_tokens=128,
        early_stopping=True,
        do_sample=True,
        top_k=50,         
        top_p=0.95,       
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    eos_token_str = tokenizer.decode([tokenizer.eos_token_id])
    chatbot_msg_split_str = "### 챗봇:"
    return tokenizer.decode(gened[0]).split(chatbot_msg_split_str)[-1].replace(eos_token_str, "")

def generate_text(text, histories, b):

    instruction = "당신은 AI 챗봇입니다. 사용자에게 도움이 되고 유익한 내용을 제공해야합니다. 답변은 길고 자세하며 친절한 설명을 덧붙여서 작성하세요.\n\n"

    for history in histories:
        human_message, chat_message  = history
        instruction += f"### 사용자:\n{human_message}\n\n"
        instruction += f"### 챗봇:\n{chat_message}\n\n"
    
    instruction += "### 사용자:\n"
    instruction += text + "\n\n"
    instruction += "### 챗봇:\n"

    return get_answer_from_chatbot(instruction)

# Gradio 인터페이스 생성 및 실행
import gradio as gr

iface = gr.ChatInterface(
        fn=generate_text,
        textbox=gr.Textbox(placeholder="placeholder", container=False, scale=7),
        title="title",
        description="description",
        examples=[["example1"], ["example2"]],
        retry_btn="다시보내기 ↩",
        undo_btn="이전채팅 삭제",
        clear_btn="전체 삭제"
)

iface.launch(share=True)


  
