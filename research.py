from ultralytics import YOLO
from utils import *

# Initializing YOLOv11 model (version 'x')
model = YOLO('yolo11x.pt')    

# Автомобиль: 0 (CVAT) -> 2 (YOLO)
yolo_to_cvat_classes_scheme = {0:2}

while True:
    print('******************************************************************************************************')
    print('Dear traveler, welcome to the research aimed at improving the lives of blind people in all of the World!')
    print('What do you desire, my friend?')
    print('1. View the classes of the standard pre-trained YOLOv11 model')
    print('2. Extract frames from a video')
    print('3. Evaluate YOLO model using pre-made annotations (done via CVAT)')
    print('4. Draw bound boxes to images')
    print('5. Walk away')

    choice = input()

    try:
        choice = int(choice)
    except ValueError:
        print("Come on! That's not even a number. Don't disappoint me. Try again, won't you? (-_-)")
        continue

    # Using class labels: bench, bus, car, person, traffic light, truck
    if choice == 1:
        classes = sorted(model.names.values())
        print("Here you go, buddy:")
        for c in classes:
            print(c)
    elif choice == 2:
        v_path = input('As you wish! Enter the path to the video:')
        o_path = input('Alright. Now enter the path to the folder where the frames should be saved (don\'t worry, buddy, I\'ll create it if it doesn\t exist):')
        print('Extracting...')
        video_to_frames(v_path, o_path)
        print('Done!')
    elif choice == 3:
        annotations_path = input('Sure! Enter the path to the folder containing right annotations: ')
        frames_path = input('Alright. Now enter the path to the folder containing all the frames: ')
        print('Calculating...')
        evaluate_yolo_model(model, frames_path, annotations_path, yolo_to_cvat_classes_scheme)
        print('Done!')
    elif choice == 4:
        labels_path = input('As you wish! Enter the path to the folder with the bounding boxes coordinates:')
        images_path = input('Alright! Now enter the path to the folder containing all the frames:')
        output_path = input('Alright! Now enter the output path:')
        print('Drawing...')
        draw_bound_boxes_to_images(labels_path, images_path, output_path)
        print('Done!')
    elif choice == 5:
        print('Farewell, dear traveler!')
        break
    else:
        print('Oh no... I didn\'t offer that option. Don\'t be upset, buddy, and try again!')