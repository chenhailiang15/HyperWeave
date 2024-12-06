from NodeCommunicate import NodeMessageReceiver



def do_action(instruct):
    print(instruct)

if __name__=="__main__":
    node_message_receiver=NodeMessageReceiver(8000)
    node_message_receiver.start_listening(do_action)

    # model1_name="ResNet18"
    # model1_ep=5
    # model1_bs=16
    # model2_name="AlexNet"
    # model2_ep=5
    # model2_bs=16
    # gpu_id=1


    