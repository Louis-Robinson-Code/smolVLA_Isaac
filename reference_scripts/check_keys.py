from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

for f in ["policy_preprocessor_step_5_normalizer_processor.safetensors",
          "policy_postprocessor_step_0_unnormalizer_processor.safetensors"]:
    print(f)
    for k, v in load_file(hf_hub_download("lerobot/smolvla_base", f)).items():
        print("  ", k, tuple(v.shape))