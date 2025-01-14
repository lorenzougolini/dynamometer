import argparse
import number_recognition

def main():
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('video_path', type=str,
                        help='A required integer positional argument')
    parser.add_argument('--plot', action='store_true',
                        help='A boolean switch')
    args = parser.parse_args()
    path_arg = args.video_path
    plot_flag = args.plot
    
    number_recognition.main(path=path_arg, plot=plot_flag)


if __name__ == "__main__":
    main()