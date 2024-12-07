import os
import matplotlib.pyplot as plt
import tensorflow as tf

# TensorBoard 로그 디렉토리
LOG_DIR = './logs'

def plot_training_logs(log_dir):
    """
    TensorBoard 로그를 기반으로 학습 및 검증의 정확도와 손실을 시각화합니다.
    Args:
        log_dir (str): TensorBoard 로그 디렉토리 경로.
    """
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    # 각 디렉토리의 이벤트 파일 처리
    train_dir = os.path.join(log_dir, 'train')
    val_dir = os.path.join(log_dir, 'validation')

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"Error: Train or validation directories do not exist in '{log_dir}'.")
        return

    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.startswith('events')]
    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.startswith('events')]

    for train_file in train_files:
        for event in tf.compat.v1.train.summary_iterator(train_file):
            for value in event.summary.value:
                if value.tag == 'epoch_loss':
                    train_loss.append(value.simple_value)
                elif value.tag == 'epoch_accuracy':
                    train_acc.append(value.simple_value)

    for val_file in val_files:
        for event in tf.compat.v1.train.summary_iterator(val_file):
            for value in event.summary.value:
                if value.tag == 'epoch_loss':
                    val_loss.append(value.simple_value)
                elif value.tag == 'epoch_accuracy':
                    val_acc.append(value.simple_value)

    if not train_loss or not val_loss:
        print("No valid data found in the logs.")
        return

    # 시각화
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 6))

    # 손실(Loss) 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 정확도(Accuracy) 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    if not os.path.exists(LOG_DIR):
        print(f"Error: Log directory '{LOG_DIR}' does not exist.")
    else:
        plot_training_logs(LOG_DIR)
