import argparse

from utils.tools import get_configs_of
# from preprocessor.preprocessor import Preprocessor
from preprocessor.preprocessor_dailytalk import Preprocessor as PreprocessorDailyTalk


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()

    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    if args.dataset == "DailyTalk":
        preprocessor = PreprocessorDailyTalk(preprocess_config, model_config, train_config)
    else:
        preprocessor = Preprocessor(preprocess_config, model_config, train_config)
    preprocessor.build_from_path()
