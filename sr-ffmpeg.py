import sys
import time
import ffmpeg
import logging
import os
import subprocess
import numpy as np
import tensorflow as tf
import DCSCN
import cv2
import re
from helper import args
from helper import utilty as util
import select
from threading import Thread
from queue import Queue, Empty

ON_POSIX = 'posix' in sys.builtin_module_names
logger = logging.getLogger(__name__)


def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()


def get_video_props(filename):
    logger.info('Getting video properties from %s' % filename)
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    s = video_info['r_frame_rate'].split("/")
    fps = float(s[0]) / float(s[1])
    duration = float(probe['format']['duration'])
    return width, height, fps, duration


def start_ffmpeg_process1(in_filename):
    logger.info('Starting ffmpeg for source file %s' % in_filename)
    args = (
        ffmpeg
        .input(in_filename)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .compile()
    )
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def start_ffmpeg_process2(in_filename, out_filename, width, height, fps, crf=20):

    logger.info('Starting ffmpeg for target file %s' % out_filename)
    args = ['ffmpeg',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%sx%s' % (width, height), '-r', str(fps), '-i', 'pipe:',
            '-i', in_filename,
            '-map', '0:V', '-map', '1', '-map', '-1:V',
            '-codec', 'copy', '-pix_fmt', 'yuv420p', '-crf', str(crf), '-preset', 'slower', '-vcodec', 'libx265',
            '-map_chapters', '1',
            '-map_metadata', '1',   # TODO Check if we need this
            '-y', out_filename]

    # args2 = (
    #     ffmpeg
    #     .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), r=fps)
    #     .output(out_filename, pix_fmt='yuv420p', vcodec='libx265', crf=crf, preset='slower')
    #     .overwrite_output()
    #     .compile()
    # )

    return subprocess.Popen(args,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, bufsize=1, close_fds=ON_POSIX)


def read_frame(process1, width, height):
    logger.debug('Reading frame')

    # Note: RGB24 == 3 bytes per pixel.
    frame_size = width * height * 3
    in_bytes = process1.stdout.read(frame_size)
    if len(in_bytes) == 0:
        frame = None
    else:
        assert len(in_bytes) == frame_size
        frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
    return frame


def write_frame(process2, frame):
    logger.debug('Writing frame')
    process2.stdin.write(
        frame
        .astype(np.uint8)
        .tobytes()
    )


def read_all_from_queue(queue: Queue, log_lines, prefix: str):
    try:
        while True:
            line = queue.get_nowait()  # or q.get(timeout=.1)
            s = line.decode(sys.stdin.encoding).rstrip()
            s = "%s %s" % (prefix, s)
            log_lines.append(s)
            logger.debug(s)
            # Do something with the output
    except Empty:
        pass  # no output yet


class FFMPEGProcessCrashException(Exception):
    pass


def run(in_filename, out_filename, process_frame):
    width, height, fps, duration = get_video_props(in_filename)

    twidth = int(FLAGS.target_width)
    theight = int(FLAGS.target_height)
    if twidth == 0 and theight == 0:
        twidth = width * int(FLAGS.scale)
        theight = height * int(FLAGS.scale)
    elif twidth == 0:
        twidth = int(width * theight / height / 2) * 2
    elif theight == 0:
        theight = int(height * twidth / width / 2) * 2

    rwidth = int(twidth / FLAGS.scale)
    rheight = int(theight / FLAGS.scale)

    uwidth = int(twidth)
    uheight = int(theight)

    # Size should be multiple of 2 (h265 needs that)
    #if uwidth % 2 == 1:
    #    uwidth += 1
    #if uheight % 2 == 1:
    #    uheight += 1

    logger.info('')
    logger.info('PLAN: ')
    logger.info('* Read frames from %s' % in_filename)
    logger.info('* Prescale frames from %dx%d to %dx%d with bicubic' % (width, height, rwidth, rheight))
    logger.info('* Upscale frames from %dx%d to %dx%d with DCSCN' % (rwidth, rheight, uwidth, uheight))
    logger.info('* Write frames to %s' % out_filename)
    logger.info('')

    process1 = start_ffmpeg_process1(in_filename)
    process2 = start_ffmpeg_process2(in_filename, out_filename, uwidth, uheight, fps)

    q1err = Queue()
    log1err = []
    p1err_t = Thread(target=enqueue_output, args=(process1.stderr, q1err))
    p1err_t.daemon = True  # thread dies with the program
    p1err_t.start()

    q2out = Queue()
    log2out = []
    p2out_t = Thread(target=enqueue_output, args=(process2.stdout, q2out))
    p2out_t.daemon = True  # thread dies with the program
    p2out_t.start()

    q2err = Queue()
    log2err = []
    p2err_t = Thread(target=enqueue_output, args=(process2.stderr, q2err))
    p2err_t.daemon = True  # thread dies with the program
    p2err_t.start()

    frame_num = 0
    moment = time.time()

    try:
        while True:
            try:
                in_frame = read_frame(process1, width, height)
                if in_frame is None:
                    logger.info('End of input stream')
                    break
                in_frame = cv2.resize(in_frame, (rwidth, rheight))
                out_frame = process_frame(in_frame)
                write_frame(process2, out_frame)

                new_moment = time.time()

                tm = float(frame_num) / fps
                hdrd = ((tm % 1.0) * 100) % 100

                secs = int(tm) % 60
                tm /= 60
                mins = tm % 60
                tm /= 60
                hrs = tm

                frame_num += 1
                total_frames = int(duration * fps)

                logger.info("Frame %d of %d (%3d%%),  Video time: %d:%02d:%02d.%02d,  Encoding FPS: %.1f" % (
                    frame_num,
                    total_frames,
                    float(frame_num) * 100 / total_frames,
                    hrs, mins, secs, hdrd,
                    1.0 / (new_moment - moment)
                ))
                moment = new_moment
            except BrokenPipeError:
                raise
            finally:
                read_all_from_queue(q1err, log1err, "in-ffmpeg-stderr>")
                read_all_from_queue(q2out, log2out, "out-ffmpeg-stdout>")
                read_all_from_queue(q2err, log2err, "out-ffmpeg-stderr>")

        logger.info('Waiting for ffmpeg process for input file')
        p1retcode = process1.wait()
        read_all_from_queue(q1err, log1err, "in-ffmpeg-stderr>")

        logger.info('Waiting for ffmpeg process for output file')
        process2.stdin.close()
        p2retcode = process2.wait()
        read_all_from_queue(q2out, log2out, "out-ffmpeg-stdout>")
        read_all_from_queue(q2err, log2err, "out-ffmpeg-stderr>")
    except BrokenPipeError:
        raise FFMPEGProcessCrashException(
            {'message': 'Pipe between broken ffmpeg and the upscaler', 'outlog': log2out, 'errlog': log1err + log2err})

    if p1retcode != 0:
        raise FFMPEGProcessCrashException(
            {'message': 'Source ffmpeg process crashed with code %d' % p1retcode, 'outlog': [], 'errlog': log1err}
        )
    if p2retcode != 0:
        raise FFMPEGProcessCrashException(
            {'message': 'Target ffmpeg process crashed with code %d' % p2retcode, 'outlog': log2out, 'errlog': log2err}
        )

    return frame_num


def parse_stdin(width, height):
    frame_size = width * height * 4
    while True:
        data = sys.stdin.buffer.read(frame_size)
        if not data: # end of input
            break
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.cvtColor(arr.reshape(height, width, 4)[:, :, 1:4], cv2.COLOR_BGR2RGB)
        yield img


ALPHA_BUF = None


def pack_image_to_stdout(img):
    global ALPHA_BUF
    arr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if ALPHA_BUF is not None:
        assert arr.shape[:-1] == ALPHA_BUF.shape[:-1]
    else:
        ALPHA_BUF = np.ones((arr.shape[0], arr.shape[1], 4), dtype=np.uint8) * 255
    ALPHA_BUF[:, :, 1:4] = arr
    sys.stdout.buffer.write(ALPHA_BUF.tobytes())


class Upscaler(DCSCN.SuperResolution):
    def __init__(self, flags, target_size=None, model_name=""):
        super().__init__(flags, model_name)
        self.target_size = target_size

    def upscale(self, org_image):
        assert len(org_image.shape) >= 3 and org_image.shape[2] == 3 and self.channels == 1

        input_y_image = util.convert_rgb_to_y(org_image)

        big_blurry_input_image = util.resize_image_by_pil(org_image, self.scale)
        big_blurry_input_ycbcr_image = util.convert_rgb_to_ycbcr(big_blurry_input_image)
        bbi_input_y_image, bbi_input_cbcr_image = util.convert_ycbcr_to_y_cbcr(big_blurry_input_ycbcr_image)

        output_y_image = self.do(input_y_image, bbi_input_y_image)

        output_image = util.convert_y_and_cbcr_to_rgb(output_y_image, bbi_input_cbcr_image)
        return output_image

'''
    def do_for_file(self, file_path, output_folder="output"):
        org_image = cv2.imread(file_path)
        output_image = self.upscale(org_image)
        target_path = os.path.basename(file_path)
        cv2.imwrite(os.path.join(output_folder, target_path), output_image)
'''


args.flags.DEFINE_string("in_file", "in.mkv", "Source video filename")
args.flags.DEFINE_string("out_file", "out.mkv", "Target video filename")
args.flags.DEFINE_string("target_width", "0", "Prescale width")
args.flags.DEFINE_string("target_height", "0", "Prescale height")
args.flags.DEFINE_boolean("debug", False, "Log DEBUG")
FLAGS = args.get()


def main(_):
    # Setting logging level
    level = logging.INFO
    if FLAGS.debug:
        level = logging.DEBUG
    logging.basicConfig(level=level)

    model = Upscaler(FLAGS, target_size=(int(FLAGS.target_width), int(FLAGS.target_height)), model_name=FLAGS.model_name)
    model.build_graph()
    model.build_optimizer()
    model.build_summary_saver()

    model.init_all_variables()
    model.load_model()

    try:
        frame_num = run(FLAGS.in_file, FLAGS.out_file, model.upscale)
        logger.info("%d frames processed succesfully" % frame_num)
        logger.info("Well done")
    except FFMPEGProcessCrashException as e:
        print("\nFatal error occured: %s\n" % e.args[0]['message'], file=sys.stderr)
        if not FLAGS.debug:
            print("Process log:")
            for log_line in e.args[0]['errlog']:
                print(log_line, file=sys.stderr)
            for log_line in e.args[0]['outlog']:
                print(log_line, file=sys.stderr)


if __name__ == '__main__':
    tf.app.run()
