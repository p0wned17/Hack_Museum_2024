import torch
from transformers import (AutoTokenizer, MBart50TokenizerFast,
                          MBartForConditionalGeneration,
                          VisionEncoderDecoderModel, ViTImageProcessor)

model = VisionEncoderDecoderModel.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning"
)
feature_extractor = ViTImageProcessor.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning"
)
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model_translator = MBartForConditionalGeneration.from_pretrained(
    "SnypzZz/Llama2-13b-Language-translate"
)
tokenizer_translator = MBart50TokenizerFast.from_pretrained(
    "SnypzZz/Llama2-13b-Language-translate", src_lang="en_XX"
)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


def predict_step(image):
    images = []
    # if image.mode != "RGB":
    i_image = image.convert(mode="RGB")

    images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


def get_pred(image):

    article_en = predict_step(image)

    model_inputs = tokenizer_translator(article_en, return_tensors="pt")
    generated_tokens = model_translator.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer_translator.lang_code_to_id["ru_RU"]
    )

    return tokenizer_translator.batch_decode(generated_tokens, skip_special_tokens=True)
