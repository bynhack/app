import time
import numpy as np
import torch
import opuslib_next
from config.logger import setup_logging
from core.providers.vad.base import VADProviderBase

TAG = __name__
logger = setup_logging()


class VADProvider(VADProviderBase):
    def __init__(self, config):
        logger.bind(tag=TAG).info("SileroVAD", config)
        self.model, self.utils = torch.hub.load(
            repo_or_dir=config["model_dir"],
            source="local",
            model="silero_vad",
            force_reload=False,
        )
        (get_speech_timestamps, _, _, _, _) = self.utils

        self.decoder = opuslib_next.Decoder(16000, 1)
        self.vad_threshold = float(config.get("threshold", 0.5))
        self.silence_threshold_ms = int(config.get("min_silence_duration_ms", 1000))

    def is_vad(self, conn, opus_packet):
        if conn.audio_type == "PCM":
            """直接处理 PCM 数据"""
            try:
                # 直接添加到缓冲区
                conn.client_audio_buffer += opus_packet

                client_have_voice = False
                chunks_processed = 0
                voice_detected_count = 0
                max_prob = 0.0

                while len(conn.client_audio_buffer) >= 512 * 2:
                    chunk = conn.client_audio_buffer[:512 * 2]
                    conn.client_audio_buffer = conn.client_audio_buffer[512 * 2:]
                    chunks_processed += 1

                    audio_int16 = np.frombuffer(chunk, dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0
                    audio_tensor = torch.from_numpy(audio_float32)

                    speech_prob = self.model(audio_tensor, 16000).item()
                    max_prob = max(max_prob, speech_prob)
                    current_have_voice = speech_prob >= self.vad_threshold
                    if current_have_voice:
                        voice_detected_count += 1
                    client_have_voice = client_have_voice or current_have_voice

                    if conn.client_have_voice and not current_have_voice:
                        stop_duration = time.time() * 1000 - conn.client_have_voice_last_time
                        if stop_duration >= self.silence_threshold_ms:
                            conn.client_voice_stop = True
                            # logger.bind(tag=TAG).info(f"检测到语音结束: 静音={stop_duration:.0f}ms")
                    if current_have_voice:
                        conn.client_have_voice = True
                        conn.client_have_voice_last_time = time.time() * 1000

                # if chunks_processed > 0:
                #     logger.bind(tag=TAG).info(
                #         f"VAD结果: {voice_detected_count}/{chunks_processed}块有声音, "
                #         f"最大概率={max_prob:.3f}"
                #     )

                return client_have_voice

            except Exception as e:
                logger.bind(tag=TAG).error(f"处理音频数据错误: {e}")
                return False
        else:
            try:
                pcm_frame = self.decoder.decode(opus_packet, 960)
                conn.client_audio_buffer += pcm_frame  # 将新数据加入缓冲区

                # 处理缓冲区中的完整帧（每次处理512采样点）
                client_have_voice = False
                while len(conn.client_audio_buffer) >= 512 * 2:
                    # 提取前512个采样点（1024字节）
                    chunk = conn.client_audio_buffer[:512 * 2]
                    conn.client_audio_buffer = conn.client_audio_buffer[512 * 2:]

                    # 转换为模型需要的张量格式
                    audio_int16 = np.frombuffer(chunk, dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0
                    audio_tensor = torch.from_numpy(audio_float32)

                    # 检测语音活动
                    speech_prob = self.model(audio_tensor, 16000).item()
                    client_have_voice = speech_prob >= self.vad_threshold


                    # 如果之前有声音，但本次没有声音，且与上次有声音的时间查已经超过了静默阈值，则认为已经说完一句话
                    if conn.client_have_voice and not client_have_voice:
                        stop_duration = time.time() * 1000 - conn.client_have_voice_last_time
                        if stop_duration >= self.silence_threshold_ms:
                            conn.client_voice_stop = True
                    if client_have_voice:
                        conn.client_have_voice = True
                        conn.client_have_voice_last_time = time.time() * 1000

                return client_have_voice
            except opuslib_next.OpusError as e:
                logger.bind(tag=TAG).info(f"解码错误: {e}")
            except Exception as e:
                logger.bind(tag=TAG).error(f"Error processing audio packet: {e}")
