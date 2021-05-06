import sys
import tensorflow as tf
import soundfile as sf
from argparse import ArgumentParser, Namespace
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import AutoProcessor
sys.path.append("TensorFlowTTS/")


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Speech synthesis')
    parser.add_argument(
        '--text',
        type=str,
        required=True
    )
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    phrase = args.text

    fastspeech2_config = AutoConfig.from_pretrained('TensorFlowTTS/examples/fastspeech/conf/fastspeech.v1.yaml')
    fastspeech2 = TFAutoModel.from_pretrained(
        config=fastspeech2_config,
        pretrained_path="models/fastspeech-150k.h5",
        name="fastspeech2"
    )

    melgan_config = AutoConfig.from_pretrained('TensorFlowTTS/examples/melgan/conf/melgan.v1.yaml')
    melgan = TFAutoModel.from_pretrained(
            config=melgan_config,
            pretrained_path="models/melgan-1M6.h5",
            name="melgan"
    )

    processor = AutoProcessor.from_pretrained(pretrained_path="ljspeech_mapper.json")
    ids = processor.text_to_sequence(phrase)
    ids = tf.expand_dims(ids, 0)

    masked_mel_before, masked_mel_after, duration_outputs = fastspeech2.inference(
        ids,
        speaker_ids=tf.zeros(shape=[tf.shape(ids)[0]], dtype=tf.int32),
        speed_ratios=tf.constant([1.0], dtype=tf.float32))

    audio_after = melgan.inference(masked_mel_after)[0, :, 0]
    sf.write('./generated.wav', audio_after, 22050, "PCM_16")