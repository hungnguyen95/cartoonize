from pytube import YouTube

def download(code):
    url = f"https://youtu.be/{code}"
    yt = YouTube(url)
    streams = yt.streams.filter(adaptive=True)
    tag_idx = streams.itag_index

    if 136 not in tag_idx or 137 not in tag_idx:
        print("Cannot find video 720p or 1080p")
        raise RuntimeError("Cannot find video 720p or 1080p")

    if 139 not in tag_idx or 140 not in tag_idx:
        print("Cannot find audio")
        raise RuntimeError("Cannot find audio")

    # Download videoraise RuntimeError()
    if 137 in tag_idx:
        video_path = streams.get_by_itag(137).download(
            output_path="./source",
            filename=code+"_video.mp4"
        )
    else:
        video_path = streams.get_by_itag(136).download(
            output_path="./source",
            filename=code+"_video.mp4"
        )

    # Download audio
    if 139 in tag_idx:
        audio_path = streams.get_by_itag(139).download(
            output_path="./source",
            filename=code+"_audio.mp4"
        )
    else:
        audio_path = streams.get_by_itag(140).download(
            output_path="./source",
            filename=code+"_audio.mp4"
        )

    return video_path, audio_path
