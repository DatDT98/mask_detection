import threading
import cv2
import time
from queue import Queue
from multiprocessing import Process, Queue, Pool
import threading, time

from detect_mask import detect
from models.experimental import attempt_load
from utils.torch_utils import select_device

def read_video(process_queue, path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    count = 0
    while True:
        ret, frame = cap.read()
        count += 1
        print("here")
        if count % 1 == 0:
            # TODO push queue
            count = 0
            process_queue.put(frame)
        if frame is None:
            print("Break read video")
            break
def process_img(img_queue, processed_queue, model,device):
    while True:
        img = img_queue.get()
        if img is not None:
            t0 = time.time()
            detect(model, img, (224,224), 0.45, 0.25, device)
            print("Time:", 1/(time.time()-t0))
            processed_queue.put(img)
        else:
            break


def show_img(processed_queue):
    while True:
        img = processed_queue.get()
        if img is not None:
            frame = cv2.resize(img,(640,640))
            print("hrer")
            cv2.imshow("Output", frame)
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            print("Break show video")
            break

def clear_queue(queue1, queue2):
    queue1.close()
    queue2.close()

def main():
    start = time.time()
    path = "person.mp4"

    # get device from file config
    device = select_device(device='cpu')
    # Load model
    model = attempt_load("yolov3.pt", map_location=device)  # load FP32 model

    process_queue = Queue()
    processed_queue = Queue()

    # with Pool(processes=4) as pool:  # start 4 worker processes
    #     result = pool.apply_async(read_video, (process_queue, path))  # evaluate "f(10)" asynchronously in a single process
    #     result = pool.apply_async(tim
    process1 = threading.Thread(target=read_video, args=(process_queue, path))
    process2 = threading.Thread(target=process_img, args=(process_queue, processed_queue, model, device))
    process3 = threading.Thread(target=show_img, args=(processed_queue,))

    # process1 = Process(target=read_video, args=(process_queue, path))
    # process2 = Process(target=process_img, args=(process_queue, processed_queue, model, device))
    # process3 = Process(target=show_img, args=(processed_queue,))

    process1.start()
    process2.start()
    process3.start()
    process1.join()
    process2.join()
    process3.join()

    clear_queue(process_queue, processed_queue)
    # with futures.ThreadPoolExecutor(max_workers=1) as executor:
    #     for i in executor.map(search, detect()):
    #         # print(i)
    #         yield i


    print("Total: {:.5f}".format(time.time() - start))


if __name__ == "__main__":
    main()
    # gen = main()
    # for value in gen:
    #     print(value)
