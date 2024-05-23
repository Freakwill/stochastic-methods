from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
from pydub import AudioSegment
from pydub.silence import detect_silence

def detect_silence_in_video(audio_path, silence_thresh=-50, min_silence_len=5000):
    print('1.1')
    audio = AudioSegment.from_file(audio_path, format="wav")
    print('1.2')
    silent_ranges = detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    return silent_ranges

def process_silence_ranges(silent_ranges, total_duration):
    processed_ranges = []
    start = 0
    for silence in silent_ranges:
        silence_start, silence_end = silence
        if silence_end - silence_start >= 10000:  # 如果静音超过10秒，保留2秒
            processed_ranges.append((silence_start / 1000, (silence_start + 2000) / 1000))
        elif silence_end - silence_start >= 5000:  # 如果静音超过5秒，保留1秒
            processed_ranges.append((silence_start / 1000, (silence_start + 1000) / 1000))
        elif silence_end - silence_start >= 2000:  # 如果静音超过2秒，保留0.2秒
            processed_ranges.append((silence_start / 1000, (silence_start + 200) / 1000))
    return processed_ranges

def create_non_silent_clips(video_path, processed_ranges, total_duration):
    video = VideoFileClip(video_path)
    non_silent_clips = []
    previous_end = 0
    for start, end in processed_ranges:
        if previous_end < start:
            non_silent_clips.append(video.subclip(previous_end, start))
        previous_end = end
    if previous_end < total_duration:
        non_silent_clips.append(video.subclip(previous_end, total_duration))
    return non_silent_clips

if __name__ == "__main__":
    video_path = "CNCE.mp4"
    audio_path = "temp_audio.wav"
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec='pcm_s16le')

    total_duration = video.duration
    print('1')
    silent_ranges = detect_silence_in_video(audio_path)
    print('2')
    processed_ranges = process_silence_ranges(silent_ranges, total_duration)
    print('3')
    non_silent_clips = create_non_silent_clips(video_path, processed_ranges, total_duration)
    print('4')
    final_clip = concatenate_videoclips(non_silent_clips)
    final_clip.write_videofile("buckup-{video_path}", codec="libx264")
