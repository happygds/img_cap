from core.rl_solve_debug import CaptioningSolver
from core.rl_model import CaptionGenerator
from core.utils import load_coco_data


def main():
  data = load_coco_data(data_path='./data', split='val')
  word_to_idx = data['word_to_idx']

  model = CaptionGenerator(word_to_idx, dim_feature=[121, 1536], dim_embed=512,
                           dim_hidden=1024, n_time_step=26, prev2out=True,
                           ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

  solver = CaptioningSolver(model, data, data, n_epochs=50, batch_size=128, update_rule='adam',
                            learning_rate=0.001, print_every=500, save_every=1, image_path='./image/',
                            pretrained_model=None, model_path='./model/rl_att_ciderD_new/',
                            test_model='./model/rl_att_ciderD_new/model-6',
                            print_bleu=True, log_path='./log/')

  # solver.save_beamsearch_result(data, split='val', save_path='./caption_eval/val_result.json', batch_size=1)
  solver.save_result(data, split='val', save_path='./caption_eval/val_result.json', batch_size=100)


if __name__ == "__main__":
  main()
