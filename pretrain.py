from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_coco_data


def main():
    # load train dataset
    data = load_coco_data(data_path='./data', split='train')
    word_to_idx = data['word_to_idx']
    # load val dataset to print out bleu scores every epoch
    val_data = load_coco_data(data_path='./data', split='val')

    model = CaptionGenerator(word_to_idx, dim_feature=[121, 1536], dim_embed=512,
                             dim_hidden=1024, n_time_step=26, prev2out=True,
                             ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

    solver = CaptioningSolver(model, data, val_data, n_epochs=25, batch_size=50, update_rule='adam',
                              learning_rate=1e-4, print_every=500, save_every=1, image_path='./image/',
                              pretrained_model=None, model_path='./model/att_jieba/',
                              n_batches=10000, print_bleu=True, log_path='./log/')

    solver.train()


if __name__ == "__main__":
    main()
