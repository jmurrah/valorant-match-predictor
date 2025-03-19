import torch
from neural_network import NeuralNetwork


def main():
    print("Hello world!")
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    nn = NeuralNetwork()
    model = nn.to(device)

    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    print("new hello")
    # pred_probab = nn.Softmax(dim=1)(logits)
    # y_pred = pred_probab.argmax(1)
    # print(f"Predicted class: {y_pred}")


if __name__ == "__main__":
    main()
