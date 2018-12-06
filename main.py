import argparse
import utils.frames as frames


def main():
    parser = argparse.ArgumentParser(description='newVidVRD')
    # Extract frames from a given video
    frame_group = parser.add_mutually_exclusive_group(required=None)
    frame_group.add_argument('-vp', '--VideoPath', type=str,
                             default='./data/VidVRD-videos/ILSVRC2015_train_00005005.mp4',
                             help="The path of Video to be extracted.")
    frame_group.add_argument('-fn', '--FramesNum', type=int, default=-1,
                             help="The number of Frames to be extracted")

    my_args = parser.parse_args()
    print(my_args)

    if my_args.VideoPath is not None:
        frames.extract_frames(my_args.VideoPath, my_args.FramesNum)


if __name__ == '__main__':
    main()
