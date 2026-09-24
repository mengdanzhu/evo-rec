import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import fire
import os


def split(input_path, output_path, cuda_list, num_samples: int = -1):
    if type(cuda_list) == int:
        cuda_list = [cuda_list]
    import data_io
    df = data_io.load_df(input_path)

    # Deterministically take the first num_samples rows when requested.
    if num_samples is not None and int(num_samples) > -1:
        num_samples = int(num_samples)
        if num_samples == 0:
            df = df.iloc[0:0]
        elif num_samples > 0:
            df = df.head(num_samples)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df_len = len(df)
    cuda_list = list(cuda_list)
    cuda_num = len(cuda_list)
    for i in range(cuda_num):
        start = i * df_len // cuda_num
        end = (i + 1) * df_len // cuda_num
        df[start:end].to_csv(f"{output_path}/{cuda_list[i]}.csv", index=True)


if __name__ == "__main__":
    fire.Fire(split)