from NodeCommunicate import NodeMessageReceiver
import json


def do_action(instruct):
    strategy_all=json.loads(instruct)
    
    
    print(strategy_all)

if __name__=="__main__":
    node_message_receiver=NodeMessageReceiver(8000)
    node_message_receiver.start_listening(do_action)