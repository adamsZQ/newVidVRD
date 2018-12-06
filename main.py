import argparse
from utils.frames import extract_frames
from utils.get_data import get_frames


def main():
    parser = argparse.ArgumentParser(description='newVidVRD')
    # Extract frames from a given video
    frame_group = parser.add_mutually_exclusive_group(required=None)
    frame_group.add_argument('-ex', '--Extract', type=bool, default=False,
                             help="Whether extract frames from a video.")
    frame_group.add_argument('-vp', '--VideoPath', type=str,
                             default='./data/VidVRD-videos/ILSVRC2015_train_00005005.mp4',
                             help="The path of Video to be extracted.")
    frame_group.add_argument('-fn', '--FramesNum', type=int, default=-1,
                             help="The number of Frames to be extracted")
    # Frame Path
    parser.add_argument('-fp', '--FramePath', type=str, default='./frames/data/VidVRD-videos/')

    my_args = parser.parse_args()
    print(my_args)

    if my_args.Extract is True:
        extract_frames(my_args.VideoPath, my_args.FramesNum)
        frames = get_frames('./frames/data/VidVRD-videos/')
    else:
        frames = get_frames(my_args.FramePath)

    print(frames)


if __name__ == '__main__':
    main()
