from NodeCommunicate import NodeMessageReceiver



def do_action(instruct):
    print(instruct)

if __name__=="__main__":
    node_message_receiver=NodeMessageReceiver("8000")
    node_message_receiver.run(do_action)