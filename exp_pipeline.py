import lark_oapi as lark

from lark_oapi.api.im.v1 import P2ImMessageReceiveV1, P2ImMessageReceiveV1Data, ReplyMessageRequest, ReplyMessageRequestBody, ReplyMessageResponse
from dotenv import load_dotenv
import os
load_dotenv()

APP_ID=os.getenv("APP_ID")
APP_SECRET=os.getenv("APP_SECRET")


from langchain_openai import ChatOpenAI
from tools.qa import QuestionAnsweringSystem, DummyAgent
qa_system = DummyAgent(chat_model=ChatOpenAI(model_name="gpt-3.5-turbo", 
                                     temperature=0, 
                                     streaming=True, 
                                     api_key=os.getenv("OPENAI_API_KEY"),
                                     base_url=os.getenv("OPENAI_BASE_URL")),
                                    #  vecstore_directory="./userdata/chroma_db_oai"
                                     )




# TODO ADD assign functions to assign msgs to different agents/check if mentions before answering
def gpt_callback(data: P2ImMessageReceiveV1):
    global qa_system
    global im_client

    msg_id=data.event.message.message_id
    data.event.message.mentions[0].name==""
    msg_content=eval(data.event.message.content)
    result=qa_system.answer_question(msg_content["text"].replace("@_user_1","You've asked the question: "))
    answer_dict={"text":result}
    request: ReplyMessageRequest = ReplyMessageRequest.builder() \
        .message_id(message_id=msg_id) \
        .request_body(ReplyMessageRequestBody.builder()
                      .content(lark.JSON.marshal(answer_dict))
                      .msg_type("text")
                      .reply_in_thread(False)
                      .build()) \
        .build()

    # 发起请求
    response: ReplyMessageResponse = im_client.im.v1.message.reply(request)
    if not response.success():
        lark.logger.error(
            f"client.im.v1.message.reply failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}")
        return

    # 处理业务结果
    lark.logger.info(lark.JSON.marshal(response.data, indent=4))
    return

def do_p2_im_message_receive_v1(data: lark.im.v1.P2ImMessageReceiveV1) -> None:
    print(f'[ do_p2_im_message_receive_v1 access ], data: {lark.JSON.marshal(data, indent=4)}')

def do_message_event(data: lark.CustomizedEvent) -> None:
    print(f'[ do_customized_event access ], type: message, data: {lark.JSON.marshal(data, indent=4)}')



event_handler = lark.EventDispatcherHandler.builder("", "") \
    .register_p2_im_message_receive_v1(gpt_callback) \
    .register_p1_customized_event("message", do_message_event) \
    .build()


cli = lark.ws.Client(APP_ID, APP_SECRET,
                        event_handler=event_handler,
                        log_level=lark.LogLevel.DEBUG)

im_client=lark.Client.builder().app_id(APP_ID) \
        .app_secret(APP_SECRET) \
        .log_level(lark.LogLevel.DEBUG) \
        .build()

cli.start()

