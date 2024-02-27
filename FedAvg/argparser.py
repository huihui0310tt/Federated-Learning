import json
import argparse


def get_arg():

    json_data = None
    with open('configure.json', 'r') as file:
        json_data = json.load(file)
    user_list = json_data['User']  # Client


    parser = argparse.ArgumentParser()
    parser.add_argument("--TaskName", type=str )
    parser.add_argument("--rounds", type=int )
    parser.add_argument("--epochs", type=int )
    parser.add_argument("--lr", type=float )
    parser.add_argument("--batch_size", type=int )
    parser.add_argument("--CUDA_Name", type=str )


    args = parser.parse_args()

    global_model = None
    # 如果需要其他外部的preTrained Model, 請在此處load進global_model

    if args.CUDA_Name == "cpu":
        no_cuda = True
    else:
        no_cuda = False


    return args.TaskName, user_list, args.rounds, args.epochs, args.lr, args.batch_size, global_model, no_cuda, args.CUDA_Name

