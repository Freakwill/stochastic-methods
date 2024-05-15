#!/usr/bin/env python

from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips

import numpy as np


def has_sound(audio_array, threshold=0.01):
    """判断音频数组是否有声音（基于音量阈值）"""
    return np.any(audio_array > threshold)


def process_clip(clip, silent_duration_threshold=2):
    """处理视频片段，删除或缩短无声部分"""
    audio = clip.audio
    frames = []
    start_time = 0
    silent_start = None
    for t in np.arange(0, clip.duration, 1/audio.fps):
        audio_frame = audio.get_frame(t)
        if has_sound(audio_frame):
            if silent_start is not None:
                # 如果当前有声音，但之前有无声部分，则处理该无声部分
                silent_duration = t - silent_start
                if silent_duration >= silent_duration_threshold:
                    # 如果无声部分超过阈值，则缩短它
                    silent_clip = clip.subclip(silent_start, t - silent_duration_threshold + 2)
                    silent_clip.set_audio(lambda t: 0)  # 设置无声
                    frames.append(silent_clip)
                silent_start = None  # 重置无声部分的起始时间
        else:
            if silent_start is None:
                silent_start = t  # 记录无声部分的起始时间
    # 添加最后一个有声部分（如果有的话）
    if silent_start is None or silent_start < clip.duration:
        frames.append(clip.subclip(silent_start if silent_start is not None else 0, clip.duration))
    # 合并所有片段
    final_clip = concatenate_videoclips(frames)
    return final_clip


if __name__ == "__main__":
    input_video = "test.mp4"  # 输入视频文件路径
    output_video = "output.mp4"  # 输出视频文件路径
    video_clip = VideoFileClip(input_video)
    processed_clip = process_clip(video_clip)
    processed_clip.write_videofile(output_video, codec='libx264')  # 保存处理后的视频
